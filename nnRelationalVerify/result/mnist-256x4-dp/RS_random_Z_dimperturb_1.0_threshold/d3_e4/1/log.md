## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 173.89956106530002


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329)
1: (-79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183)
2: (-104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471)
3: (-110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219)
4: (-101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509)
5: (-90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867)
6: (-86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223)
7: (-95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773)
8: (-114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580)
9: (-86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.47 + 10.65 = 12.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -174.0736347, upper bound: 174.0736347

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0736347, upper bound: 174.0736347
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0736347, upper bound: 174.0736347
time: 7.48 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.02 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.02
Output dim: 7, lower bound: -174.0736347, upper bound: 174.0736347
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.02
Output dim: 7, lower bound: -174.0736347, upper bound: 174.0736347

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0425191, upper bound: 174.0425175
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0425191, upper bound: 174.0425175
time: 6.02 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457609, upper bound: 173.9457616
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457609, upper bound: 173.9457616
time: 7.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.01 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.01
Output dim: 7, lower bound: -174.0425191, upper bound: 174.0425175
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.01
Output dim: 7, lower bound: -174.0425191, upper bound: 174.0425175
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.01
Output dim: 7, lower bound: -173.9457609, upper bound: 173.9457616
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.01
Output dim: 7, lower bound: -173.9457609, upper bound: 173.9457616

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 139

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9770788, upper bound: 173.9770788
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9770788, upper bound: 173.9770788
time: 6.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0363028, upper bound: 174.0363021
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0363025, upper bound: 174.0363022
time: 6.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457603, upper bound: 173.9457616
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457609, upper bound: 173.9457609
time: 7.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457479, upper bound: 173.9457486
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457479, upper bound: 173.9457486
time: 5.85 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 17.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.40
Output dim: 7, lower bound: -173.9770788, upper bound: 173.9770788
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.40
Output dim: 7, lower bound: -173.9770788, upper bound: 173.9770788
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.40
Output dim: 7, lower bound: -174.0363028, upper bound: 174.0363021
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.40
Output dim: 7, lower bound: -174.0363025, upper bound: 174.0363022
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.40
Output dim: 7, lower bound: -173.9457603, upper bound: 173.9457616
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.40
Output dim: 7, lower bound: -173.9457609, upper bound: 173.9457609
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.40
Output dim: 7, lower bound: -173.9457479, upper bound: 173.9457486
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.40
Output dim: 7, lower bound: -173.9457479, upper bound: 173.9457486

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9770788, upper bound: 173.9770737
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9770733, upper bound: 173.9770788
time: 6.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9735744, upper bound: 173.9735816
time: 6.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9735818, upper bound: 173.9735745
time: 6.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9940272, upper bound: 173.9940241
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9940272, upper bound: 173.9940241
time: 6.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0362945, upper bound: 174.0362919
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0362921, upper bound: 174.0362943
time: 6.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9321827, upper bound: 173.9321869
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9321827, upper bound: 173.9321869
time: 7.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457609, upper bound: 173.9457606
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457596, upper bound: 173.9457609
time: 6.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457456, upper bound: 173.9457486
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457479, upper bound: 173.9457448
time: 6.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457479, upper bound: 173.9457480
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457475, upper bound: 173.9457486
time: 5.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 7, lower bound: -173.9770788, upper bound: 173.9770737
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 7, lower bound: -173.9770733, upper bound: 173.9770788
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 7, lower bound: -173.9735744, upper bound: 173.9735816
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 7, lower bound: -173.9735818, upper bound: 173.9735745
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 7, lower bound: -173.9940272, upper bound: 173.9940241
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 7, lower bound: -173.9940272, upper bound: 173.9940241
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 7, lower bound: -174.0362945, upper bound: 174.0362919
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 7, lower bound: -174.0362921, upper bound: 174.0362943
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 7, lower bound: -173.9321827, upper bound: 173.9321869
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 7, lower bound: -173.9321827, upper bound: 173.9321869
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 7, lower bound: -173.9457609, upper bound: 173.9457606
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 7, lower bound: -173.9457596, upper bound: 173.9457609
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 7, lower bound: -173.9457456, upper bound: 173.9457486
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 7, lower bound: -173.9457479, upper bound: 173.9457448
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 7, lower bound: -173.9457479, upper bound: 173.9457480
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 7, lower bound: -173.9457475, upper bound: 173.9457486

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9734557, upper bound: 173.9734481
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9734511, upper bound: 173.9734493
time: 6.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 91

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9472684, upper bound: 173.9472669
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9472684, upper bound: 173.9472669
time: 6.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9735736, upper bound: 173.9735816
time: 6.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9735744, upper bound: 173.9735816
time: 6.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9701318, upper bound: 173.9701313
time: 6.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9701318, upper bound: 173.9701313
time: 6.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9429487, upper bound: 173.9429515
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9429487, upper bound: 173.9429515
time: 6.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 73

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9859517, upper bound: 173.9859628
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9859517, upper bound: 173.9859628
time: 5.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0356172, upper bound: 174.0356149
time: 8.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0356172, upper bound: 174.0356149
time: 7.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0334699, upper bound: 174.0334806
time: 7.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0334699, upper bound: 174.0334806
time: 7.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9313586, upper bound: 173.9313635
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9313592, upper bound: 173.9313625
time: 5.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9321820, upper bound: 173.9321867
time: 7.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9321827, upper bound: 173.9321865
time: 7.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9444966, upper bound: 173.9445006
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9445037, upper bound: 173.9444994
time: 6.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457594, upper bound: 173.9457609
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457596, upper bound: 173.9457608
time: 6.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457456, upper bound: 173.9457486
time: 7.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457455, upper bound: 173.9457467
time: 5.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457332, upper bound: 173.9457321
time: 6.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9457352, upper bound: 173.9457314
time: 5.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9380510, upper bound: 173.9380509
time: 6.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9380527, upper bound: 173.9380496
time: 6.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 139
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9420381, upper bound: 173.9420432
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9420421, upper bound: 173.9420392
time: 5.90 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 12.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9734557, upper bound: 173.9734481
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9734511, upper bound: 173.9734493
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9472684, upper bound: 173.9472669
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9472684, upper bound: 173.9472669
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9735736, upper bound: 173.9735816
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9735744, upper bound: 173.9735816
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9701318, upper bound: 173.9701313
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9701318, upper bound: 173.9701313
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9429487, upper bound: 173.9429515
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9429487, upper bound: 173.9429515
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9859517, upper bound: 173.9859628
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9859517, upper bound: 173.9859628
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -174.0356172, upper bound: 174.0356149
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -174.0356172, upper bound: 174.0356149
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -174.0334699, upper bound: 174.0334806
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -174.0334699, upper bound: 174.0334806
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9313586, upper bound: 173.9313635
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9313592, upper bound: 173.9313625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9321820, upper bound: 173.9321867
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9321827, upper bound: 173.9321865
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9444966, upper bound: 173.9445006
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9445037, upper bound: 173.9444994
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9457594, upper bound: 173.9457609
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9457596, upper bound: 173.9457608
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9457456, upper bound: 173.9457486
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9457455, upper bound: 173.9457467
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9457332, upper bound: 173.9457321
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9457352, upper bound: 173.9457314
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9380510, upper bound: 173.9380509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9380527, upper bound: 173.9380496
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9420381, upper bound: 173.9420432
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.86
Output dim: 7, lower bound: -173.9420421, upper bound: 173.9420392

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9734550, upper bound: 173.9734473
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9734544, upper bound: 173.9734472
time: 6.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9699619, upper bound: 173.9699669
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9699619, upper bound: 173.9699669
time: 6.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9433760, upper bound: 173.9433759
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9433719, upper bound: 173.9433761
time: 6.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9472684, upper bound: 173.9472669
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9472678, upper bound: 173.9472669
time: 5.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=61, inp2_unstable=61, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=188, inp2_unstable=188, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 75
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 207
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 91
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9735733, upper bound: 173.9735816
time: 8.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9735736, upper bound: 173.9735812
time: 6.91 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 21.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.11
Output dim: 7, lower bound: -173.9734550, upper bound: 173.9734473
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.11
Output dim: 7, lower bound: -173.9734544, upper bound: 173.9734472
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.11
Output dim: 7, lower bound: -173.9699619, upper bound: 173.9699669
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.11
Output dim: 7, lower bound: -173.9699619, upper bound: 173.9699669
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.11
Output dim: 7, lower bound: -173.9433760, upper bound: 173.9433759
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.11
Output dim: 7, lower bound: -173.9433719, upper bound: 173.9433761
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.11
Output dim: 7, lower bound: -173.9472684, upper bound: 173.9472669
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.11
Output dim: 7, lower bound: -173.9472678, upper bound: 173.9472669
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 21.11
Output dim: 7, lower bound: -173.9735733, upper bound: 173.9735816
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 21.11
Output dim: 7, lower bound: -173.9735736, upper bound: 173.9735812
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9735744, upper bound: 173.9735816
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9701318, upper bound: 173.9701313
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9701318, upper bound: 173.9701313
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9429487, upper bound: 173.9429515
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9429487, upper bound: 173.9429515
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9859517, upper bound: 173.9859628
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9859517, upper bound: 173.9859628
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -174.0356172, upper bound: 174.0356149
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -174.0356172, upper bound: 174.0356149
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -174.0334699, upper bound: 174.0334806
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -174.0334699, upper bound: 174.0334806
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9313586, upper bound: 173.9313635
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9313592, upper bound: 173.9313625
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9321820, upper bound: 173.9321867
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9321827, upper bound: 173.9321865
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9444966, upper bound: 173.9445006
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9445037, upper bound: 173.9444994
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9457594, upper bound: 173.9457609
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9457596, upper bound: 173.9457608
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9457456, upper bound: 173.9457486
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9457455, upper bound: 173.9457467
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9457332, upper bound: 173.9457321
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9457352, upper bound: 173.9457314
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9380510, upper bound: 173.9380509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9380527, upper bound: 173.9380496
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9420381, upper bound: 173.9420432
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.11
Output dim: 7, lower bound: -173.9420421, upper bound: 173.9420392

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 12.12 + 595.07 = 607.18 seconds
