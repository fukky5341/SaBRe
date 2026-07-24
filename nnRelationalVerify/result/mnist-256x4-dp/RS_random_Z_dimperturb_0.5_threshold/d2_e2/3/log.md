## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0125892


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012595, 0.0012595)
1: (-0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039169, 0.0039169)
2: (0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0058660, 0.0058660)
3: (-0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0043302, 0.0043302)
4: (-0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687)
5: (0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0043145, 0.0043145)
6: (0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620)
7: (-0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0086060, 0.0086060)
8: (0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0273852, 0.0273852)
9: (0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0074588, 0.0074588)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.16 + 1.77 = 2.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0175321, upper bound: 0.0175321

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0174816, upper bound: 0.0173476
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0173476, upper bound: 0.0174816
time: 0.89 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.89 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.89
Output dim: 8, lower bound: -0.0174816, upper bound: 0.0173476
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.89
Output dim: 8, lower bound: -0.0173476, upper bound: 0.0174816

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012547, 0.0012511
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039169, 0.0039169
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0058293, 0.0058543
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0042981, 0.0043169
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0042824, 0.0043012
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0085585, 0.0085178
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0273852, 0.0273852
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0073869, 0.0074212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0164958, upper bound: 0.0168278
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169433, upper bound: 0.0163610
time: 0.84 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012511, 0.0012547
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039169, 0.0039169
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0058543, 0.0058293
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0043169, 0.0042981
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0043012, 0.0042824
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0085178, 0.0085585
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0273852, 0.0273852
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0074212, 0.0073869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0163517, upper bound: 0.0164281
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0163517, upper bound: 0.0164281
time: 0.99 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 8, lower bound: -0.0164958, upper bound: 0.0168278
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 8, lower bound: -0.0169433, upper bound: 0.0163610
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 8, lower bound: -0.0163517, upper bound: 0.0164281
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 8, lower bound: -0.0163517, upper bound: 0.0164281

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012104, 0.0012147
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038506, 0.0038707
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0056020, 0.0055720
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041247, 0.0041021
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041088, 0.0040863
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080659, 0.0081149
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0262136, 0.0263538
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070628, 0.0070216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0161632, upper bound: 0.0164464
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0161489, upper bound: 0.0165137
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012178, 0.0012069
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038853, 0.0038339
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055470, 0.0056239
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040833, 0.0041411
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040675, 0.0041253
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081506, 0.0080252
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0264561, 0.0260970
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069874, 0.0070929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0167573, upper bound: 0.0162733
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0168514, upper bound: 0.0162137
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012415, 0.0012443
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039169, 0.0039169
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0057779, 0.0057580
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0042613, 0.0042464
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0042457, 0.0042308
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0084065, 0.0084389
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0270744, 0.0271674
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0073180, 0.0072907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0161995, upper bound: 0.0162767
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0161725, upper bound: 0.0162845
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012511, 0.0012450
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039169, 0.0039169
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0057830, 0.0058293
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0042652, 0.0042981
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0042496, 0.0042824
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0085178, 0.0084472
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0273852, 0.0271911
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0073250, 0.0073869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0162118, upper bound: 0.0163903
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0163133, upper bound: 0.0162755
time: 0.97 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.02 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 8, lower bound: -0.0161632, upper bound: 0.0164464
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 8, lower bound: -0.0161489, upper bound: 0.0165137
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 8, lower bound: -0.0167573, upper bound: 0.0162733
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 8, lower bound: -0.0168514, upper bound: 0.0162137
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 8, lower bound: -0.0161995, upper bound: 0.0162767
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 8, lower bound: -0.0161725, upper bound: 0.0162845
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 8, lower bound: -0.0162118, upper bound: 0.0163903
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 8, lower bound: -0.0163133, upper bound: 0.0162755

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011946, 0.0011950
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038031, 0.0038050
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055046, 0.0055016
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040538, 0.0040515
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040384, 0.0040362
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079227, 0.0079275
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0258837, 0.0258976
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069191, 0.0069150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159834, upper bound: 0.0163454
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160709, upper bound: 0.0162616
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011908, 0.0012013
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037850, 0.0038342
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055483, 0.0054746
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040866, 0.0040312
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040712, 0.0040159
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078786, 0.0079987
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0257574, 0.0261014
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069790, 0.0068779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159106, upper bound: 0.0162691
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159129, upper bound: 0.0162564
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012162, 0.0012055
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038771, 0.0038272
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055369, 0.0056116
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040757, 0.0041319
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040599, 0.0041160
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081305, 0.0080087
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0263985, 0.0260497
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069734, 0.0070760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151529, upper bound: 0.0148640
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151529, upper bound: 0.0148640
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012165, 0.0012051
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038785, 0.0038253
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055340, 0.0056138
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040735, 0.0041335
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040578, 0.0041177
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081340, 0.0080041
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0264087, 0.0260363
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069695, 0.0070790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0165828, upper bound: 0.0159771
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0165824, upper bound: 0.0159771
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012232, 0.0012215
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039169, 0.0039104
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0056678, 0.0056798
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041796, 0.0041886
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041642, 0.0041732
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0082433, 0.0082238
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0267073, 0.0266514
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0071581, 0.0071746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160689, upper bound: 0.0161887
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0161158, upper bound: 0.0161388
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012187, 0.0012250
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038971, 0.0039169
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0056923, 0.0056479
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041980, 0.0041646
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041826, 0.0041492
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081913, 0.0082636
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0265584, 0.0267656
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0071917, 0.0071308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159273, upper bound: 0.0160362
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159276, upper bound: 0.0160362
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012410, 0.0012359
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039169, 0.0039169
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0057294, 0.0057699
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0042239, 0.0042527
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0042083, 0.0042370
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0084018, 0.0083412
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0271361, 0.0269446
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0072429, 0.0072971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157166, upper bound: 0.0158152
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157166, upper bound: 0.0158152
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012421, 0.0012349
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039169, 0.0039169
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0057223, 0.0057779
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0042186, 0.0042587
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0042030, 0.0042431
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0084149, 0.0083297
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0271736, 0.0269115
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0072332, 0.0073081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160530, upper bound: 0.0160254
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160530, upper bound: 0.0160254
time: 0.84 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 8, lower bound: -0.0159834, upper bound: 0.0163454
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 8, lower bound: -0.0160709, upper bound: 0.0162616
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 8, lower bound: -0.0159106, upper bound: 0.0162691
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 8, lower bound: -0.0159129, upper bound: 0.0162564
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 8, lower bound: -0.0151529, upper bound: 0.0148640
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 8, lower bound: -0.0151529, upper bound: 0.0148640
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 8, lower bound: -0.0165828, upper bound: 0.0159771
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 8, lower bound: -0.0165824, upper bound: 0.0159771
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 8, lower bound: -0.0160689, upper bound: 0.0161887
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 8, lower bound: -0.0161158, upper bound: 0.0161388
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 8, lower bound: -0.0159273, upper bound: 0.0160362
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 8, lower bound: -0.0159276, upper bound: 0.0160362
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 8, lower bound: -0.0157166, upper bound: 0.0158152
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 8, lower bound: -0.0157166, upper bound: 0.0158152
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 8, lower bound: -0.0160530, upper bound: 0.0160254
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 8, lower bound: -0.0160530, upper bound: 0.0160254

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011931, 0.0011937
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037952, 0.0037982
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054943, 0.0054898
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040460, 0.0040426
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040307, 0.0040273
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079034, 0.0079108
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0258285, 0.0258496
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069050, 0.0068988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159083, upper bound: 0.0163041
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159344, upper bound: 0.0160322
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011933, 0.0011934
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037962, 0.0037966
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054919, 0.0054914
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040442, 0.0040438
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040289, 0.0040284
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079059, 0.0079068
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0258357, 0.0258382
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069017, 0.0069009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142329, upper bound: 0.0144356
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142329, upper bound: 0.0144356
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011874, 0.0011974
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037693, 0.0038162
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055213, 0.0054510
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040663, 0.0040135
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040509, 0.0039982
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078402, 0.0079547
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0256473, 0.0259754
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069420, 0.0068456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157410, upper bound: 0.0161792
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0158185, upper bound: 0.0161130
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011869, 0.0012013
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037670, 0.0038342
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055483, 0.0054476
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040866, 0.0040109
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040712, 0.0039956
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078346, 0.0079987
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0256313, 0.0261014
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069790, 0.0068409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156652, upper bound: 0.0159320
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156621, upper bound: 0.0159352
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012105, 0.0012051
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038541, 0.0038288
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055375, 0.0055753
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040760, 0.0041045
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040604, 0.0040888
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080677, 0.0080060
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0262300, 0.0260533
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069717, 0.0070236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150017, upper bound: 0.0147129
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150113, upper bound: 0.0147129
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012162, 0.0011998
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038771, 0.0038042
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055006, 0.0056116
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040483, 0.0041319
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040327, 0.0041160
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081305, 0.0079459
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0263985, 0.0258811
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069211, 0.0070760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149994, upper bound: 0.0148334
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151254, upper bound: 0.0148099
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012114, 0.0011971
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038513, 0.0037843
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054726, 0.0055730
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040274, 0.0041028
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040117, 0.0040870
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080676, 0.0079040
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0262183, 0.0257497
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068853, 0.0070230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0162279, upper bound: 0.0159322
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0165480, upper bound: 0.0158896
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012084, 0.0012051
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038375, 0.0038253
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055340, 0.0055524
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040735, 0.0040873
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040578, 0.0040716
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080340, 0.0080041
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0261221, 0.0260363
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069695, 0.0069947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152929, upper bound: 0.0148531
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152929, upper bound: 0.0148531
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012216, 0.0012202
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039103, 0.0039035
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0056575, 0.0056677
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041719, 0.0041795
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041565, 0.0041641
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0082236, 0.0082070
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0266509, 0.0266034
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0071440, 0.0071580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159583, upper bound: 0.0161475
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160253, upper bound: 0.0160073
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012219, 0.0012199
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039115, 0.0039020
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0056553, 0.0056695
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041702, 0.0041809
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041548, 0.0041655
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0082265, 0.0082033
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0266593, 0.0265928
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0071409, 0.0071605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155150, upper bound: 0.0155581
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155150, upper bound: 0.0155581
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012006, 0.0012088
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038087, 0.0038473
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055711, 0.0055135
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041056, 0.0040622
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040902, 0.0040469
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079789, 0.0080728
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0259354, 0.0262046
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070238, 0.0069447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157470, upper bound: 0.0157670
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156479, upper bound: 0.0158767
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012014, 0.0012069
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038123, 0.0038384
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055578, 0.0055188
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040956, 0.0040662
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040802, 0.0040509
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079876, 0.0080512
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0259604, 0.0261425
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070056, 0.0069520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144034, upper bound: 0.0144568
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144034, upper bound: 0.0144568
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012381, 0.0012321
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039169, 0.0039169
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0057024, 0.0057492
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0042036, 0.0042371
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041880, 0.0042215
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0083681, 0.0082972
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0270395, 0.0268184
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0072058, 0.0072687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151236, upper bound: 0.0152751
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151236, upper bound: 0.0152751
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012372, 0.0012359
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039169, 0.0039169
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0057294, 0.0057428
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0042239, 0.0042324
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0042083, 0.0042167
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0083577, 0.0083412
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0270098, 0.0269446
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0072429, 0.0072600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156407, upper bound: 0.0157804
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156743, upper bound: 0.0157481
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012245, 0.0012174
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038982, 0.0038622
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055885, 0.0056430
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041169, 0.0041569
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041015, 0.0041412
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0082158, 0.0081345
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0265447, 0.0262880
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070672, 0.0071373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159439, upper bound: 0.0159626
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159849, upper bound: 0.0159378
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012257, 0.0012164
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039037, 0.0038576
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055816, 0.0056513
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041118, 0.0041631
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040964, 0.0041475
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0082293, 0.0081233
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0265834, 0.0262560
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070578, 0.0071487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157180, upper bound: 0.0157015
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157058, upper bound: 0.0157070
time: 1.01 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0159083, upper bound: 0.0163041
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0159344, upper bound: 0.0160322
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0142329, upper bound: 0.0144356
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0142329, upper bound: 0.0144356
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0157410, upper bound: 0.0161792
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0158185, upper bound: 0.0161130
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0156652, upper bound: 0.0159320
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0156621, upper bound: 0.0159352
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0150017, upper bound: 0.0147129
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0150113, upper bound: 0.0147129
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0149994, upper bound: 0.0148334
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0151254, upper bound: 0.0148099
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0162279, upper bound: 0.0159322
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0165480, upper bound: 0.0158896
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0152929, upper bound: 0.0148531
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0152929, upper bound: 0.0148531
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0159583, upper bound: 0.0161475
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0160253, upper bound: 0.0160073
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0155150, upper bound: 0.0155581
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0155150, upper bound: 0.0155581
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0157470, upper bound: 0.0157670
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0156479, upper bound: 0.0158767
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0144034, upper bound: 0.0144568
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0144034, upper bound: 0.0144568
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0151236, upper bound: 0.0152751
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0151236, upper bound: 0.0152751
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0156407, upper bound: 0.0157804
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0156743, upper bound: 0.0157481
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0159439, upper bound: 0.0159626
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0159849, upper bound: 0.0159378
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0157180, upper bound: 0.0157015
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 8, lower bound: -0.0157058, upper bound: 0.0157070

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011846, 0.0011886
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037655, 0.0037847
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054717, 0.0054430
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040276, 0.0040061
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040122, 0.0039907
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078121, 0.0078588
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0256115, 0.0257453
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068639, 0.0068246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156436, upper bound: 0.0159065
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154960, upper bound: 0.0160236
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011858, 0.0011852
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037712, 0.0037686
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054475, 0.0054515
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040095, 0.0040125
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039941, 0.0039971
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078260, 0.0078194
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0256513, 0.0256326
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068308, 0.0068363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153239, upper bound: 0.0153900
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153239, upper bound: 0.0153887
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011906, 0.0011962
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037846, 0.0038109
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055133, 0.0054738
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040604, 0.0040307
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040449, 0.0040153
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078762, 0.0079405
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0257540, 0.0259382
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069306, 0.0068764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140390, upper bound: 0.0142379
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140390, upper bound: 0.0142379
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011933, 0.0011907
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037962, 0.0037849
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054743, 0.0054914
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040311, 0.0040438
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040157, 0.0040284
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079059, 0.0078771
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0258357, 0.0257565
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068772, 0.0069009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140390, upper bound: 0.0142379
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140390, upper bound: 0.0142379
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011857, 0.0011961
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037606, 0.0038094
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055112, 0.0054380
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040587, 0.0040037
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040433, 0.0039884
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078191, 0.0079382
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0255868, 0.0259281
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069281, 0.0068278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154912, upper bound: 0.0157946
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152987, upper bound: 0.0159083
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011861, 0.0011956
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037625, 0.0038069
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055074, 0.0054409
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040558, 0.0040058
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040405, 0.0039906
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078237, 0.0079320
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0256000, 0.0259105
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069229, 0.0068317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157267, upper bound: 0.0160772
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157789, upper bound: 0.0158882
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011812, 0.0011934
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037360, 0.0037936
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054874, 0.0054012
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040408, 0.0039760
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040255, 0.0039608
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077590, 0.0078995
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0254148, 0.0258172
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068955, 0.0067772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154269, upper bound: 0.0155563
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152343, upper bound: 0.0156650
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011792, 0.0012013
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037264, 0.0038342
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055483, 0.0053868
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040866, 0.0039652
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040712, 0.0039500
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077355, 0.0079987
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0253475, 0.0261014
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069790, 0.0067574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152385, upper bound: 0.0154917
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152317, upper bound: 0.0155051
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011949, 0.0011910
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037725, 0.0037543
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054312, 0.0054585
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039996, 0.0040201
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039843, 0.0040047
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078853, 0.0078408
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0256791, 0.0255518
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068316, 0.0068690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147363, upper bound: 0.0144116
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147145, upper bound: 0.0144235
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012000, 0.0011895
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037962, 0.0037473
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054206, 0.0054939
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039916, 0.0040467
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039763, 0.0040313
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079430, 0.0078236
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0258445, 0.0255024
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068171, 0.0069176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145059, upper bound: 0.0142012
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145059, upper bound: 0.0142012
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012073, 0.0011923
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038477, 0.0037812
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054613, 0.0055618
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040160, 0.0040927
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040003, 0.0040767
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080271, 0.0078573
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0261710, 0.0257031
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068515, 0.0069925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148320, upper bound: 0.0146832
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148549, upper bound: 0.0146832
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012108, 0.0011911
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038643, 0.0037759
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054532, 0.0055867
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040100, 0.0041114
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039943, 0.0040955
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080677, 0.0078442
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0262874, 0.0256656
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068404, 0.0070267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148569, upper bound: 0.0144961
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148341, upper bound: 0.0145131
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012024, 0.0011892
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038212, 0.0037594
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054296, 0.0055221
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039933, 0.0040629
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039775, 0.0040470
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079625, 0.0078117
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0259858, 0.0255538
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068111, 0.0069380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150290, upper bound: 0.0148069
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150290, upper bound: 0.0148069
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012056, 0.0011881
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038360, 0.0037542
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054218, 0.0055443
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039874, 0.0040795
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039716, 0.0040636
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079985, 0.0077989
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0260891, 0.0255172
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068003, 0.0069684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140119, upper bound: 0.0134901
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140119, upper bound: 0.0134901
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012054, 0.0012048
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038248, 0.0038254
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055337, 0.0055329
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040735, 0.0040728
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040578, 0.0040571
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079990, 0.0080005
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0260308, 0.0260349
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069670, 0.0069658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151268, upper bound: 0.0145795
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149961, upper bound: 0.0146524
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012084, 0.0012021
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038375, 0.0038125
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055145, 0.0055524
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040590, 0.0040873
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040433, 0.0040716
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080340, 0.0079691
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0261221, 0.0259450
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069406, 0.0069947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151268, upper bound: 0.0145795
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149961, upper bound: 0.0146524
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012114, 0.0012113
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038703, 0.0038698
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0056054, 0.0056061
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041319, 0.0041324
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041165, 0.0041170
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081098, 0.0081087
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0263652, 0.0263622
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070655, 0.0070664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145756, upper bound: 0.0146829
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145756, upper bound: 0.0146829
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012122, 0.0012099
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038741, 0.0038635
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055959, 0.0056119
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041248, 0.0041368
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041094, 0.0041214
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081193, 0.0080932
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0263923, 0.0263177
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070524, 0.0070743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156943, upper bound: 0.0156751
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156715, upper bound: 0.0156903
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012179, 0.0012120
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038887, 0.0038615
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055947, 0.0056355
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041246, 0.0041553
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041093, 0.0041399
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081710, 0.0081046
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0265003, 0.0263100
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070578, 0.0071137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149907, upper bound: 0.0149895
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149783, upper bound: 0.0150091
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012141, 0.0012199
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038710, 0.0039020
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0056553, 0.0056089
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041702, 0.0041353
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041548, 0.0041200
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081278, 0.0082033
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0263764, 0.0265928
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0071409, 0.0070773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149907, upper bound: 0.0149895
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149783, upper bound: 0.0150091
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011809, 0.0011805
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037633, 0.0037617
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054362, 0.0054386
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040022, 0.0040040
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039868, 0.0039886
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078043, 0.0078004
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0255896, 0.0255786
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068170, 0.0068202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155905, upper bound: 0.0156746
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156366, upper bound: 0.0156211
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011723, 0.0011902
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037232, 0.0038070
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055041, 0.0053785
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040532, 0.0039588
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040378, 0.0039436
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077065, 0.0079110
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0253093, 0.0258954
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069101, 0.0067379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151121, upper bound: 0.0153599
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151121, upper bound: 0.0153599
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011985, 0.0012074
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038012, 0.0038430
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055645, 0.0055019
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041000, 0.0040529
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040845, 0.0040375
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079554, 0.0080574
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0258811, 0.0261732
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070118, 0.0069259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142559, upper bound: 0.0142325
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141556, upper bound: 0.0143342
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012014, 0.0012041
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038123, 0.0038273
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055409, 0.0055188
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040823, 0.0040662
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040668, 0.0040509
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079876, 0.0080190
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0259604, 0.0260633
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069795, 0.0069520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142946, upper bound: 0.0144083
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143544, upper bound: 0.0143517
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012327, 0.0012239
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039169, 0.0038947
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0056402, 0.0057067
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041568, 0.0042052
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041413, 0.0041896
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0082988, 0.0081958
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0268410, 0.0265281
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0071205, 0.0072104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149513, upper bound: 0.0150874
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149516, upper bound: 0.0150874
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012299, 0.0012321
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039169, 0.0039169
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0057024, 0.0056870
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0042036, 0.0041904
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041880, 0.0041748
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0082668, 0.0082972
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0267492, 0.0268184
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0072058, 0.0071834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149513, upper bound: 0.0150874
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149516, upper bound: 0.0150874
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012354, 0.0012345
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039169, 0.0039169
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0057193, 0.0057300
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0042163, 0.0042227
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0042007, 0.0042071
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0083368, 0.0083247
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0269497, 0.0268973
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0072290, 0.0072423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154765, upper bound: 0.0156029
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154034, upper bound: 0.0156659
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012358, 0.0012342
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039169, 0.0039169
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0057170, 0.0057328
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0042146, 0.0042248
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041989, 0.0042092
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0083414, 0.0083209
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0269631, 0.0268865
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0072258, 0.0072462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148104, upper bound: 0.0151432
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150250, upper bound: 0.0148524
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012227, 0.0012160
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038893, 0.0038554
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055783, 0.0056298
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041093, 0.0041469
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040939, 0.0041313
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081943, 0.0081179
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0264831, 0.0262406
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070532, 0.0071192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157766, upper bound: 0.0156840
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156842, upper bound: 0.0157980
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012231, 0.0012158
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038914, 0.0038543
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055766, 0.0056328
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041080, 0.0041492
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040926, 0.0041336
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081992, 0.0081151
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0264973, 0.0262327
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070509, 0.0071234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0158239, upper bound: 0.0157381
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157897, upper bound: 0.0157648
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012221, 0.0012113
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038876, 0.0038346
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055472, 0.0056271
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040858, 0.0041450
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040705, 0.0041294
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081900, 0.0080672
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0264707, 0.0260952
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070105, 0.0071156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155826, upper bound: 0.0155279
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155566, upper bound: 0.0155449
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012206, 0.0012164
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038807, 0.0038576
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055816, 0.0056168
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041118, 0.0041372
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040964, 0.0041216
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081732, 0.0081233
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0264226, 0.0262560
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070578, 0.0071014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147968, upper bound: 0.0150409
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150973, upper bound: 0.0148782
time: 1.07 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0156436, upper bound: 0.0159065
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0154960, upper bound: 0.0160236
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0153239, upper bound: 0.0153900
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0153239, upper bound: 0.0153887
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0140390, upper bound: 0.0142379
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0140390, upper bound: 0.0142379
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0140390, upper bound: 0.0142379
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0140390, upper bound: 0.0142379
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0154912, upper bound: 0.0157946
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0152987, upper bound: 0.0159083
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0157267, upper bound: 0.0160772
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0157789, upper bound: 0.0158882
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0154269, upper bound: 0.0155563
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0152343, upper bound: 0.0156650
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0152385, upper bound: 0.0154917
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0152317, upper bound: 0.0155051
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0147363, upper bound: 0.0144116
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0147145, upper bound: 0.0144235
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0145059, upper bound: 0.0142012
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0145059, upper bound: 0.0142012
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0148320, upper bound: 0.0146832
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0148549, upper bound: 0.0146832
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0148569, upper bound: 0.0144961
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0148341, upper bound: 0.0145131
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0150290, upper bound: 0.0148069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0150290, upper bound: 0.0148069
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0140119, upper bound: 0.0134901
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0140119, upper bound: 0.0134901
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0151268, upper bound: 0.0145795
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0149961, upper bound: 0.0146524
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0151268, upper bound: 0.0145795
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0149961, upper bound: 0.0146524
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0145756, upper bound: 0.0146829
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0145756, upper bound: 0.0146829
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0156943, upper bound: 0.0156751
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0156715, upper bound: 0.0156903
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0149907, upper bound: 0.0149895
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0149783, upper bound: 0.0150091
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0149907, upper bound: 0.0149895
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0149783, upper bound: 0.0150091
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0155905, upper bound: 0.0156746
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0156366, upper bound: 0.0156211
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0151121, upper bound: 0.0153599
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0151121, upper bound: 0.0153599
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0142559, upper bound: 0.0142325
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0141556, upper bound: 0.0143342
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0142946, upper bound: 0.0144083
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0143544, upper bound: 0.0143517
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0149513, upper bound: 0.0150874
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0149516, upper bound: 0.0150874
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0149513, upper bound: 0.0150874
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0149516, upper bound: 0.0150874
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0154765, upper bound: 0.0156029
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0154034, upper bound: 0.0156659
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0148104, upper bound: 0.0151432
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0150250, upper bound: 0.0148524
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0157766, upper bound: 0.0156840
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0156842, upper bound: 0.0157980
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0158239, upper bound: 0.0157381
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0157897, upper bound: 0.0157648
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0155826, upper bound: 0.0155279
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0155566, upper bound: 0.0155449
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0147968, upper bound: 0.0150409
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.06
Output dim: 8, lower bound: -0.0150973, upper bound: 0.0148782

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011656, 0.0011607
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037346, 0.0037117
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053536, 0.0053878
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039365, 0.0039622
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039208, 0.0039465
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076597, 0.0076039
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0253625, 0.0252027
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066742, 0.0067212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143624, upper bound: 0.0143524
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143624, upper bound: 0.0143524
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011566, 0.0011696
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036926, 0.0037534
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054160, 0.0053250
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039834, 0.0039149
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039676, 0.0038993
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0075572, 0.0077055
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0250689, 0.0254939
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067598, 0.0066349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153350, upper bound: 0.0158979
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153346, upper bound: 0.0158408
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011829, 0.0011868
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037591, 0.0037772
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054602, 0.0054332
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040191, 0.0039988
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040036, 0.0039834
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077955, 0.0078395
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0255663, 0.0256922
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068483, 0.0068113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151482, upper bound: 0.0152271
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151473, upper bound: 0.0152053
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011858, 0.0011824
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037712, 0.0037565
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054292, 0.0054515
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039958, 0.0040125
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039804, 0.0039971
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078260, 0.0077890
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0256513, 0.0255476
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068058, 0.0068363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150638, upper bound: 0.0149712
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149560, upper bound: 0.0151344
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011743, 0.0011841
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037052, 0.0037513
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054244, 0.0053555
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039927, 0.0039409
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039774, 0.0039256
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076940, 0.0078063
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0252027, 0.0255244
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068173, 0.0067227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0132597, upper bound: 0.0134196
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0132597, upper bound: 0.0134196
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011757, 0.0011799
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037121, 0.0037316
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053950, 0.0053657
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039706, 0.0039486
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039553, 0.0039333
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077106, 0.0077583
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0252503, 0.0253869
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067769, 0.0067367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0136673, upper bound: 0.0138519
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0136673, upper bound: 0.0138609
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011771, 0.0011795
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037169, 0.0037296
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053920, 0.0053737
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039684, 0.0039546
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039531, 0.0039394
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077246, 0.0077535
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0252863, 0.0253732
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067729, 0.0067470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139831, upper bound: 0.0141946
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139951, upper bound: 0.0140915
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011785, 0.0011744
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037237, 0.0037056
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053561, 0.0053839
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039413, 0.0039623
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039261, 0.0039471
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077413, 0.0076949
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0253339, 0.0252052
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067235, 0.0067610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0132596, upper bound: 0.0134067
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0132596, upper bound: 0.0134067
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011674, 0.0011686
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037321, 0.0037375
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053956, 0.0053875
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039691, 0.0039630
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039537, 0.0039476
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076738, 0.0076871
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0253576, 0.0253956
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067419, 0.0067307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154158, upper bound: 0.0157579
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154489, upper bound: 0.0155265
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011581, 0.0011764
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036887, 0.0037740
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054502, 0.0053225
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040102, 0.0039141
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039947, 0.0038988
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0075679, 0.0077761
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0250543, 0.0256505
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068168, 0.0066416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152153, upper bound: 0.0158718
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152550, upper bound: 0.0156331
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011774, 0.0011900
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037334, 0.0037922
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054829, 0.0053948
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040361, 0.0039699
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040207, 0.0039545
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077335, 0.0078771
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0253865, 0.0257978
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068794, 0.0067585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155573, upper bound: 0.0159422
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155575, upper bound: 0.0158692
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011784, 0.0011869
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037378, 0.0037778
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054613, 0.0054014
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040199, 0.0039748
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040044, 0.0039595
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077443, 0.0078419
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0254173, 0.0256969
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068497, 0.0067675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155303, upper bound: 0.0154983
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153251, upper bound: 0.0156198
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011627, 0.0011657
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037069, 0.0037211
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053710, 0.0053497
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039506, 0.0039346
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039352, 0.0039193
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076123, 0.0076470
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0251815, 0.0252807
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067082, 0.0066790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153275, upper bound: 0.0155212
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153872, upper bound: 0.0153314
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011535, 0.0011735
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036635, 0.0037575
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054256, 0.0052848
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039917, 0.0038858
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039762, 0.0038705
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0075065, 0.0077360
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0248782, 0.0255357
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067831, 0.0065899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148521, upper bound: 0.0152483
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148281, upper bound: 0.0152742
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011757, 0.0011963
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037112, 0.0038115
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055143, 0.0053640
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040610, 0.0039480
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040456, 0.0039329
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076984, 0.0079433
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0252411, 0.0259427
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069324, 0.0067262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150926, upper bound: 0.0153789
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150930, upper bound: 0.0153362
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011741, 0.0012013
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037038, 0.0038342
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055483, 0.0053529
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040866, 0.0039397
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040712, 0.0039245
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076803, 0.0079987
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0251894, 0.0261014
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069790, 0.0067110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150152, upper bound: 0.0151845
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148280, upper bound: 0.0152742
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011805, 0.0011708
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037357, 0.0036902
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053328, 0.0054010
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039245, 0.0039758
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039093, 0.0039605
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077649, 0.0076538
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0254146, 0.0250962
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066882, 0.0067817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119297, upper bound: 0.0116361
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119297, upper bound: 0.0116361
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011747, 0.0011769
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037084, 0.0037189
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053759, 0.0053601
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039569, 0.0039450
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039416, 0.0039297
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076982, 0.0077240
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0252235, 0.0252974
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067473, 0.0067256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141352, upper bound: 0.0138577
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141352, upper bound: 0.0138585
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011959, 0.0011844
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037765, 0.0037230
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053843, 0.0054644
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039643, 0.0040245
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039490, 0.0040091
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078949, 0.0077644
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0257066, 0.0253327
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067672, 0.0068771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142601, upper bound: 0.0138025
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141234, upper bound: 0.0139407
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011949, 0.0011895
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037719, 0.0037473
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054206, 0.0054575
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039916, 0.0040194
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039763, 0.0040040
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078838, 0.0078236
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0256748, 0.0255024
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068171, 0.0068677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142601, upper bound: 0.0138025
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141234, upper bound: 0.0139410
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011916, 0.0011778
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037636, 0.0037037
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053491, 0.0054401
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039349, 0.0040040
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039195, 0.0039885
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078475, 0.0076914
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0255979, 0.0251748
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067108, 0.0068393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138999, upper bound: 0.0136997
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138999, upper bound: 0.0136997
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011967, 0.0011766
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037873, 0.0036981
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053408, 0.0054755
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039287, 0.0040307
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039133, 0.0040151
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079052, 0.0076779
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0257634, 0.0251361
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066994, 0.0068879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143690, upper bound: 0.0141678
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143690, upper bound: 0.0141678
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011969, 0.0011719
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038235, 0.0037098
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053584, 0.0055299
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039428, 0.0040714
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039275, 0.0040559
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079536, 0.0076704
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0260170, 0.0252171
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067058, 0.0069438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0136223, upper bound: 0.0131065
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0136223, upper bound: 0.0131065
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011916, 0.0011765
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037984, 0.0037311
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053903, 0.0054922
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039668, 0.0040431
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039515, 0.0040276
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078922, 0.0077223
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0258410, 0.0253659
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067495, 0.0068921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146872, upper bound: 0.0143723
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146953, upper bound: 0.0143721
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011993, 0.0011889
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038081, 0.0037595
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054295, 0.0055023
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039929, 0.0040477
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039772, 0.0040318
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079268, 0.0078082
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0258943, 0.0255544
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068083, 0.0069082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148412, upper bound: 0.0145615
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148015, upper bound: 0.0145961
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012024, 0.0011861
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038212, 0.0037463
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054098, 0.0055221
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039781, 0.0040629
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039624, 0.0040470
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079625, 0.0077761
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0259858, 0.0254623
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067812, 0.0069380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148368, upper bound: 0.0146358
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148756, upper bound: 0.0146359
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012001, 0.0011877
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038141, 0.0037563
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054240, 0.0055104
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039880, 0.0040530
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039723, 0.0040372
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079374, 0.0077966
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0259326, 0.0255290
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068003, 0.0069189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0137129, upper bound: 0.0130275
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0135449, upper bound: 0.0131237
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012056, 0.0011826
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038360, 0.0037323
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053879, 0.0055443
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039609, 0.0040795
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039452, 0.0040636
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079985, 0.0077378
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0260891, 0.0253607
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067508, 0.0069684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0137129, upper bound: 0.0130275
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0135449, upper bound: 0.0131237
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011902, 0.0011795
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037917, 0.0037469
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054097, 0.0054768
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039796, 0.0040300
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039642, 0.0040146
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078665, 0.0077572
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0257750, 0.0254618
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067818, 0.0068739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147596, upper bound: 0.0142098
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147548, upper bound: 0.0142231
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011803, 0.0011887
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037449, 0.0037901
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054744, 0.0054068
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040283, 0.0039774
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040128, 0.0039620
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077524, 0.0078626
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0254480, 0.0257639
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068706, 0.0067778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147255, upper bound: 0.0146055
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149560, upper bound: 0.0145922
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011932, 0.0011767
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038041, 0.0037340
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053905, 0.0054956
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039651, 0.0040448
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039498, 0.0040293
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078996, 0.0077258
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0258628, 0.0253719
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067554, 0.0069005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149486, upper bound: 0.0144088
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149722, upper bound: 0.0144088
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011832, 0.0011854
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037573, 0.0037746
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054513, 0.0054256
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040109, 0.0039922
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039955, 0.0039767
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077855, 0.0078250
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0255358, 0.0256560
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068389, 0.0068044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148151, upper bound: 0.0144806
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148461, upper bound: 0.0144808
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012085, 0.0012107
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038581, 0.0038684
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0056032, 0.0055877
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041299, 0.0041183
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041144, 0.0041028
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080765, 0.0081017
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0262796, 0.0263519
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070603, 0.0070390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142828, upper bound: 0.0144408
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142814, upper bound: 0.0144408
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012114, 0.0012084
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038703, 0.0038577
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055871, 0.0056061
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041178, 0.0041324
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041023, 0.0041170
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081098, 0.0080754
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0263652, 0.0262766
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070381, 0.0070664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142828, upper bound: 0.0144408
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142814, upper bound: 0.0144408
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012090, 0.0012051
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038598, 0.0038413
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055627, 0.0055905
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040998, 0.0041207
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040845, 0.0041053
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080843, 0.0080391
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0262923, 0.0261628
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070069, 0.0070449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154541, upper bound: 0.0154486
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154600, upper bound: 0.0154484
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012073, 0.0012099
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038520, 0.0038635
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055959, 0.0055787
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041248, 0.0041118
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041094, 0.0040965
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080652, 0.0080932
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0262375, 0.0263177
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070524, 0.0070288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143488, upper bound: 0.0143305
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143488, upper bound: 0.0143305
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012140, 0.0012070
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038717, 0.0038388
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055607, 0.0056100
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040990, 0.0041361
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040837, 0.0041208
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081295, 0.0080492
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0263814, 0.0261511
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070111, 0.0070788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148754, upper bound: 0.0149453
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149464, upper bound: 0.0148663
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012128, 0.0012120
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038660, 0.0038615
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055947, 0.0056014
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041246, 0.0041297
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041093, 0.0041143
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081156, 0.0081046
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0263414, 0.0263100
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070578, 0.0070670

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148710, upper bound: 0.0149650
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149337, upper bound: 0.0148749
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012105, 0.0012149
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038553, 0.0038793
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0056214, 0.0055853
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041447, 0.0041176
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041293, 0.0041023
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080894, 0.0081481
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0262663, 0.0264346
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070944, 0.0070450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144237, upper bound: 0.0144120
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144237, upper bound: 0.0144120
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012090, 0.0012199
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038483, 0.0039020
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0056553, 0.0055749
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041702, 0.0041097
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041548, 0.0040944
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080724, 0.0082033
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0262175, 0.0265928
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0071409, 0.0070306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148710, upper bound: 0.0149650
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149337, upper bound: 0.0148749
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011793, 0.0011792
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037554, 0.0037552
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054264, 0.0054267
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039948, 0.0039951
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039795, 0.0039797
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077850, 0.0077845
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0255344, 0.0255329
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068036, 0.0068040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154649, upper bound: 0.0156312
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155450, upper bound: 0.0154790
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011795, 0.0011789
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037568, 0.0037539
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054245, 0.0054288
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039934, 0.0039966
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039781, 0.0039813
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077884, 0.0077813
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0255439, 0.0255238
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068009, 0.0068068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154928, upper bound: 0.0155772
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155918, upper bound: 0.0154572
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011669, 0.0011822
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036944, 0.0037661
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054428, 0.0053355
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040071, 0.0039264
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039918, 0.0039112
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076363, 0.0078111
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0251082, 0.0256092
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068260, 0.0066788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149225, upper bound: 0.0153202
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150720, upper bound: 0.0151767
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011643, 0.0011902
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036823, 0.0038070
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055041, 0.0053172
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040532, 0.0039127
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040378, 0.0038976
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076066, 0.0079110
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0250231, 0.0258954
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069101, 0.0066538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146016, upper bound: 0.0148394
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145830, upper bound: 0.0148485
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011794, 0.0011791
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037581, 0.0037566
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054286, 0.0054309
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039964, 0.0039981
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039811, 0.0039828
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077897, 0.0077860
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0255534, 0.0255427
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068063, 0.0068094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141162, upper bound: 0.0141842
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142071, upper bound: 0.0140961
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011702, 0.0011877
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037148, 0.0037969
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054890, 0.0053660
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040418, 0.0039493
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040264, 0.0039341
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076841, 0.0078844
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0252506, 0.0258246
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068892, 0.0067205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138481, upper bound: 0.0140395
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138460, upper bound: 0.0140397
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011914, 0.0011951
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037726, 0.0037920
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054867, 0.0054576
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040410, 0.0040192
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040255, 0.0040038
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078753, 0.0079179
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0256721, 0.0258085
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068979, 0.0068606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128016, upper bound: 0.0129274
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128016, upper bound: 0.0129274
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011930, 0.0011940
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037799, 0.0037867
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054787, 0.0054685
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040350, 0.0040275
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040196, 0.0040121
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078932, 0.0079049
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0257233, 0.0257714
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068870, 0.0068757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0125630, upper bound: 0.0125279
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0125630, upper bound: 0.0125279
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012151, 0.0012068
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038506, 0.0038092
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055092, 0.0055717
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040573, 0.0041033
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040420, 0.0040878
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080997, 0.0080052
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0262121, 0.0259177
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069583, 0.0070396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143662, upper bound: 0.0144628
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143662, upper bound: 0.0144632
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012157, 0.0012055
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038532, 0.0038028
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054995, 0.0055756
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040500, 0.0041062
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040347, 0.0040907
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081060, 0.0079895
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0262302, 0.0258726
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069451, 0.0070449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143662, upper bound: 0.0144628
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143662, upper bound: 0.0144632
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012123, 0.0012150
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038375, 0.0038507
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055714, 0.0055521
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041040, 0.0040885
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040887, 0.0040730
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080676, 0.0081066
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0261203, 0.0262080
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070436, 0.0070126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143662, upper bound: 0.0144628
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143662, upper bound: 0.0144632
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012125, 0.0012136
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038382, 0.0038443
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055617, 0.0055532
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040968, 0.0040894
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040814, 0.0040739
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080695, 0.0080908
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0261257, 0.0261629
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070304, 0.0070142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148008, upper bound: 0.0149620
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147783, upper bound: 0.0149739
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012186, 0.0012093
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0039154, 0.0038708
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0056005, 0.0056683
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041251, 0.0041766
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041095, 0.0041609
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081992, 0.0080928
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0266631, 0.0263467
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0070528, 0.0071428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131423, upper bound: 0.0132446
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131423, upper bound: 0.0132446
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012099, 0.0012180
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038748, 0.0039112
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0056610, 0.0056074
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0041706, 0.0041309
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0041549, 0.0041153
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0081000, 0.0081915
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0263790, 0.0266292
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0071359, 0.0070593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151446, upper bound: 0.0153676
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151446, upper bound: 0.0153676
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011928, 0.0012007
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037798, 0.0038144
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055117, 0.0054602
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040556, 0.0040163
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040398, 0.0040005
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078615, 0.0079502
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0256965, 0.0259369
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069277, 0.0068530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146238, upper bound: 0.0149400
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145605, upper bound: 0.0150139
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012007, 0.0011907
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038165, 0.0037677
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054418, 0.0055152
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040031, 0.0040576
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039873, 0.0040418
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079511, 0.0078363
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0259534, 0.0256107
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068318, 0.0069285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126161, upper bound: 0.0124744
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126161, upper bound: 0.0124744
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012060, 0.0011914
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038454, 0.0037773
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054579, 0.0055604
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040166, 0.0040945
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040013, 0.0040790
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080508, 0.0078830
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0261621, 0.0256831
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068667, 0.0070132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148290, upper bound: 0.0150185
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151826, upper bound: 0.0148102
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011974, 0.0012009
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038055, 0.0038221
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055249, 0.0055005
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040670, 0.0040495
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040516, 0.0040341
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079533, 0.0079921
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0258827, 0.0259959
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069586, 0.0069311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153817, upper bound: 0.0154745
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153505, upper bound: 0.0154945
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012057, 0.0011936
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038400, 0.0037824
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054722, 0.0055578
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040302, 0.0040944
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040148, 0.0040789
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080412, 0.0078991
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0261421, 0.0257403
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068807, 0.0070022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153138, upper bound: 0.0152274
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153138, upper bound: 0.0152274
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012013, 0.0011980
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038196, 0.0038029
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0055029, 0.0055272
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040533, 0.0040713
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040379, 0.0040560
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079913, 0.0079492
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0259992, 0.0258838
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0069228, 0.0069602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148547, upper bound: 0.0151142
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151802, upper bound: 0.0148989
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012050, 0.0011893
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038377, 0.0037634
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054438, 0.0055542
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040089, 0.0040917
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039935, 0.0040763
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0080354, 0.0078529
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0261255, 0.0256079
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068418, 0.0069973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148623, upper bound: 0.0148376
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148623, upper bound: 0.0148376
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0012005, 0.0011932
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0038165, 0.0037815
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054709, 0.0055226
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040292, 0.0040679
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040138, 0.0040525
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0079838, 0.0078970
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0259777, 0.0257342
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068789, 0.0069539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153883, upper bound: 0.0152492
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152957, upper bound: 0.0153722
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011795, 0.0011828
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037084, 0.0037198
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053747, 0.0053574
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039563, 0.0039418
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039409, 0.0039264
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077127, 0.0077439
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0252119, 0.0252921
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067510, 0.0067258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146608, upper bound: 0.0149807
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147123, upper bound: 0.0149494
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011933, 0.0011754
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037727, 0.0036852
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053228, 0.0054537
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039172, 0.0040143
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039020, 0.0039987
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078697, 0.0076592
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0256615, 0.0250497
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066797, 0.0068580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143377, upper bound: 0.0141640
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143377, upper bound: 0.0141640
time: 0.90 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0143624, upper bound: 0.0143524
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0143624, upper bound: 0.0143524
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0153350, upper bound: 0.0158979
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0153346, upper bound: 0.0158408
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0151482, upper bound: 0.0152271
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0151473, upper bound: 0.0152053
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0150638, upper bound: 0.0149712
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0149560, upper bound: 0.0151344
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0132597, upper bound: 0.0134196
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0132597, upper bound: 0.0134196
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0136673, upper bound: 0.0138519
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0136673, upper bound: 0.0138609
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0139831, upper bound: 0.0141946
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0139951, upper bound: 0.0140915
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0132596, upper bound: 0.0134067
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0132596, upper bound: 0.0134067
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0154158, upper bound: 0.0157579
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0154489, upper bound: 0.0155265
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0152153, upper bound: 0.0158718
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0152550, upper bound: 0.0156331
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0155573, upper bound: 0.0159422
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0155575, upper bound: 0.0158692
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0155303, upper bound: 0.0154983
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0153251, upper bound: 0.0156198
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0153275, upper bound: 0.0155212
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0153872, upper bound: 0.0153314
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148521, upper bound: 0.0152483
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148281, upper bound: 0.0152742
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0150926, upper bound: 0.0153789
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0150930, upper bound: 0.0153362
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0150152, upper bound: 0.0151845
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148280, upper bound: 0.0152742
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0119297, upper bound: 0.0116361
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0119297, upper bound: 0.0116361
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0141352, upper bound: 0.0138577
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0141352, upper bound: 0.0138585
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0142601, upper bound: 0.0138025
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0141234, upper bound: 0.0139407
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0142601, upper bound: 0.0138025
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0141234, upper bound: 0.0139410
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0138999, upper bound: 0.0136997
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0138999, upper bound: 0.0136997
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0143690, upper bound: 0.0141678
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0143690, upper bound: 0.0141678
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0136223, upper bound: 0.0131065
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0136223, upper bound: 0.0131065
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0146872, upper bound: 0.0143723
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0146953, upper bound: 0.0143721
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148412, upper bound: 0.0145615
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148015, upper bound: 0.0145961
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148368, upper bound: 0.0146358
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148756, upper bound: 0.0146359
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0137129, upper bound: 0.0130275
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0135449, upper bound: 0.0131237
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0137129, upper bound: 0.0130275
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0135449, upper bound: 0.0131237
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0147596, upper bound: 0.0142098
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0147548, upper bound: 0.0142231
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0147255, upper bound: 0.0146055
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0149560, upper bound: 0.0145922
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0149486, upper bound: 0.0144088
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0149722, upper bound: 0.0144088
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148151, upper bound: 0.0144806
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148461, upper bound: 0.0144808
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0142828, upper bound: 0.0144408
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0142814, upper bound: 0.0144408
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0142828, upper bound: 0.0144408
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0142814, upper bound: 0.0144408
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0154541, upper bound: 0.0154486
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0154600, upper bound: 0.0154484
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0143488, upper bound: 0.0143305
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0143488, upper bound: 0.0143305
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148754, upper bound: 0.0149453
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0149464, upper bound: 0.0148663
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148710, upper bound: 0.0149650
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0149337, upper bound: 0.0148749
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0144237, upper bound: 0.0144120
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0144237, upper bound: 0.0144120
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148710, upper bound: 0.0149650
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0149337, upper bound: 0.0148749
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0154649, upper bound: 0.0156312
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0155450, upper bound: 0.0154790
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0154928, upper bound: 0.0155772
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0155918, upper bound: 0.0154572
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0149225, upper bound: 0.0153202
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0150720, upper bound: 0.0151767
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0146016, upper bound: 0.0148394
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0145830, upper bound: 0.0148485
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0141162, upper bound: 0.0141842
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0142071, upper bound: 0.0140961
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0138481, upper bound: 0.0140395
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0138460, upper bound: 0.0140397
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0128016, upper bound: 0.0129274
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0128016, upper bound: 0.0129274
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0125630, upper bound: 0.0125279
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0125630, upper bound: 0.0125279
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0143662, upper bound: 0.0144628
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0143662, upper bound: 0.0144632
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0143662, upper bound: 0.0144628
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0143662, upper bound: 0.0144632
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0143662, upper bound: 0.0144628
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0143662, upper bound: 0.0144632
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148008, upper bound: 0.0149620
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0147783, upper bound: 0.0149739
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0131423, upper bound: 0.0132446
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0131423, upper bound: 0.0132446
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0151446, upper bound: 0.0153676
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0151446, upper bound: 0.0153676
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0146238, upper bound: 0.0149400
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0145605, upper bound: 0.0150139
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0126161, upper bound: 0.0124744
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0126161, upper bound: 0.0124744
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148290, upper bound: 0.0150185
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0151826, upper bound: 0.0148102
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0153817, upper bound: 0.0154745
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0153505, upper bound: 0.0154945
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0153138, upper bound: 0.0152274
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0153138, upper bound: 0.0152274
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148547, upper bound: 0.0151142
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0151802, upper bound: 0.0148989
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148623, upper bound: 0.0148376
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0148623, upper bound: 0.0148376
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0153883, upper bound: 0.0152492
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0152957, upper bound: 0.0153722
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0146608, upper bound: 0.0149807
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0147123, upper bound: 0.0149494
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0143377, upper bound: 0.0141640
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.02
Output dim: 8, lower bound: -0.0143377, upper bound: 0.0141640

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011601, 0.0011656
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037123, 0.0037381
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053932, 0.0053545
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039662, 0.0039371
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039505, 0.0039214
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0075980, 0.0076611
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0252055, 0.0253864
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067246, 0.0066715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130199, upper bound: 0.0130621
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130199, upper bound: 0.0130621
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011656, 0.0011552
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037346, 0.0036894
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053202, 0.0053878
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039114, 0.0039622
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0038957, 0.0039465
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076597, 0.0075422
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0253625, 0.0250457
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066245, 0.0067212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142041, upper bound: 0.0142271
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142052, upper bound: 0.0142099
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011408, 0.0011597
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036133, 0.0037017
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053403, 0.0052080
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039267, 0.0038272
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039111, 0.0038118
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0073769, 0.0075925
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0245217, 0.0251394
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066586, 0.0064771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151102, upper bound: 0.0155545
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149985, upper bound: 0.0156740
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011423, 0.0011538
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036202, 0.0036741
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0052990, 0.0052184
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0038956, 0.0038350
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0038801, 0.0038195
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0073938, 0.0075252
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0245700, 0.0249466
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066020, 0.0064913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147226, upper bound: 0.0151834
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147226, upper bound: 0.0151834
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011668, 0.0011760
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036788, 0.0037218
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053783, 0.0053140
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039584, 0.0039100
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039432, 0.0038950
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076122, 0.0077170
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0250097, 0.0253099
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067444, 0.0066561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0134259, upper bound: 0.0135386
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0134259, upper bound: 0.0135386
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011680, 0.0011707
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036842, 0.0036968
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053409, 0.0053221
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039303, 0.0039161
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039152, 0.0039010
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076253, 0.0076561
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0250474, 0.0251356
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066931, 0.0066672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145223, upper bound: 0.0145743
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145223, upper bound: 0.0145743
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011664, 0.0011545
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037384, 0.0036834
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053115, 0.0053936
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039049, 0.0039665
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0038893, 0.0039508
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076690, 0.0075352
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0253893, 0.0250051
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066159, 0.0067291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147716, upper bound: 0.0146700
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147344, upper bound: 0.0146812
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011578, 0.0011641
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036983, 0.0037282
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053785, 0.0053335
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039553, 0.0039213
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039396, 0.0039057
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0075711, 0.0076444
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0251087, 0.0253180
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067079, 0.0066466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143685, upper bound: 0.0146088
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143685, upper bound: 0.0146088
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011696, 0.0011763
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036795, 0.0037108
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053639, 0.0053169
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039472, 0.0039119
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039319, 0.0038967
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076311, 0.0077076
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0250226, 0.0252416
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067342, 0.0066698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0132003, upper bound: 0.0133774
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0132161, upper bound: 0.0132964
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011665, 0.0011841
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036648, 0.0037513
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054244, 0.0052949
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039927, 0.0038953
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039774, 0.0038802
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0075953, 0.0078063
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0249199, 0.0255244
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068173, 0.0066396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130176, upper bound: 0.0130734
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129257, upper bound: 0.0131717
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011720, 0.0011749
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036953, 0.0037090
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053611, 0.0053406
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039451, 0.0039297
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039298, 0.0039145
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076697, 0.0077030
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0251332, 0.0252286
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067304, 0.0067023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127017, upper bound: 0.0128707
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127017, upper bound: 0.0128707
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011707, 0.0011799
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036894, 0.0037316
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053950, 0.0053318
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039706, 0.0039231
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039553, 0.0039079
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076554, 0.0077583
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0250921, 0.0253869
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067769, 0.0066902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0136149, upper bound: 0.0138155
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0136207, upper bound: 0.0137418
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011686, 0.0011740
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036864, 0.0037133
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053647, 0.0053251
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039480, 0.0039178
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039328, 0.0039027
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076308, 0.0076932
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0250615, 0.0252462
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067254, 0.0066710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0136081, upper bound: 0.0138175
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0136081, upper bound: 0.0138259
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011699, 0.0011710
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036927, 0.0036992
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053436, 0.0053346
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039321, 0.0039249
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039169, 0.0039097
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076462, 0.0076588
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0251056, 0.0251476
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066964, 0.0066839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0136135, upper bound: 0.0137558
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0136135, upper bound: 0.0137580
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011736, 0.0011665
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036968, 0.0036651
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0052955, 0.0053435
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0038957, 0.0039320
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0038806, 0.0039168
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076755, 0.0075961
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0251454, 0.0249223
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066404, 0.0067056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127017, upper bound: 0.0128707
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127013, upper bound: 0.0128707
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011707, 0.0011744
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036833, 0.0037056
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053561, 0.0053233
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039413, 0.0039168
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039261, 0.0039016
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076425, 0.0076949
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0250511, 0.0252052
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067235, 0.0066779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130175, upper bound: 0.0130633
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129256, upper bound: 0.0131607
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011586, 0.0011628
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037020, 0.0037218
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053687, 0.0053390
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039478, 0.0039255
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039321, 0.0039098
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0075801, 0.0076285
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0251343, 0.0252730
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066949, 0.0066541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144437, upper bound: 0.0147332
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144437, upper bound: 0.0147332
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011590, 0.0011597
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037042, 0.0037074
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053471, 0.0053423
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039316, 0.0039280
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039159, 0.0039123
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0075855, 0.0075933
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0251499, 0.0251723
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066653, 0.0066587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152163, upper bound: 0.0152438
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152163, upper bound: 0.0152599
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011493, 0.0011702
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036586, 0.0037564
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054206, 0.0052740
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039868, 0.0038766
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039710, 0.0038610
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0074742, 0.0077130
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0248310, 0.0255152
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067661, 0.0065650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149313, upper bound: 0.0155116
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148593, upper bound: 0.0155674
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011503, 0.0011675
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036633, 0.0037439
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054017, 0.0052811
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039726, 0.0038819
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039569, 0.0038664
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0074858, 0.0076823
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0248641, 0.0254273
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067402, 0.0065747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149690, upper bound: 0.0153285
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148844, upper bound: 0.0153466
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011613, 0.0011784
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036532, 0.0037336
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053957, 0.0052754
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039709, 0.0038804
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039556, 0.0038653
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0075498, 0.0077458
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0248293, 0.0253909
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067678, 0.0066027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153175, upper bound: 0.0156485
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153148, upper bound: 0.0156492
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011618, 0.0011738
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036559, 0.0037121
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053635, 0.0052794
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039466, 0.0038834
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039315, 0.0038683
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0075563, 0.0076933
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0248480, 0.0252406
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067236, 0.0066082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144335, upper bound: 0.0146823
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144335, upper bound: 0.0146823
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011592, 0.0011592
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037047, 0.0037049
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053433, 0.0053431
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039287, 0.0039286
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039131, 0.0039129
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0075868, 0.0075872
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0251538, 0.0251547
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066601, 0.0066599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153624, upper bound: 0.0153557
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153665, upper bound: 0.0153113
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011506, 0.0011672
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036649, 0.0037425
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053996, 0.0052834
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039711, 0.0038837
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039553, 0.0038681
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0074896, 0.0076789
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0248751, 0.0254174
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067373, 0.0065779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150936, upper bound: 0.0153446
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150917, upper bound: 0.0153530
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011537, 0.0011597
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036771, 0.0037052
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053438, 0.0053017
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039291, 0.0038974
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039134, 0.0038818
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0075193, 0.0075880
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0249604, 0.0251570
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066608, 0.0066030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151687, upper bound: 0.0154038
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151713, upper bound: 0.0153531
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011549, 0.0011567
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036827, 0.0036908
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053222, 0.0053102
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039129, 0.0039038
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0038973, 0.0038882
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0075331, 0.0075528
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0249998, 0.0250562
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066312, 0.0066146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139096, upper bound: 0.0139699
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139096, upper bound: 0.0139699
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011500, 0.0011685
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036482, 0.0037353
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053922, 0.0052619
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039666, 0.0038686
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039512, 0.0038533
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0074692, 0.0076816
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0247713, 0.0253799
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067373, 0.0065584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0094743, upper bound: 0.0096106
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0094743, upper bound: 0.0096106
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011485, 0.0011735
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036411, 0.0037575
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054256, 0.0052513
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039917, 0.0038606
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039762, 0.0038454
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0074519, 0.0077360
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0247217, 0.0255357
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067831, 0.0065439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147324, upper bound: 0.0152368
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147877, upper bound: 0.0150999
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011594, 0.0011847
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036319, 0.0037542
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054296, 0.0052464
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039967, 0.0038589
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039814, 0.0038438
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0075171, 0.0078157
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0246917, 0.0255471
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068237, 0.0065723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0135419, upper bound: 0.0138746
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0135419, upper bound: 0.0138746
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011603, 0.0011800
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036358, 0.0037322
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053966, 0.0052522
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039719, 0.0038633
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039566, 0.0038482
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0075267, 0.0077620
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0247191, 0.0253932
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067784, 0.0065803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0134788, upper bound: 0.0136495
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0134788, upper bound: 0.0136495
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011550, 0.0011737
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036718, 0.0037620
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054323, 0.0052972
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039967, 0.0038951
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039812, 0.0038798
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0075267, 0.0077469
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0249362, 0.0255669
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067923, 0.0066069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149418, upper bound: 0.0151482
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149735, upper bound: 0.0149723
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011464, 0.0011815
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036315, 0.0037985
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0054869, 0.0052368
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0040378, 0.0038497
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0040222, 0.0038345
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0074284, 0.0078359
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0246543, 0.0258219
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0068672, 0.0065241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146843, upper bound: 0.0151595
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146842, upper bound: 0.0151255
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011714, 0.0011719
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036937, 0.0036963
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053420, 0.0053381
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039314, 0.0039285
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039162, 0.0039133
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076625, 0.0076688
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0251212, 0.0251391
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067008, 0.0066955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0102970, upper bound: 0.0101653
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0102970, upper bound: 0.0101653
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011697, 0.0011769
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036857, 0.0037189
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053759, 0.0053262
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039569, 0.0039195
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039416, 0.0039043
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076430, 0.0077240
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0250653, 0.0252974
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067473, 0.0066791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0102970, upper bound: 0.0101653
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0102970, upper bound: 0.0101653
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011819, 0.0011592
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037517, 0.0036457
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0052591, 0.0054178
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0038676, 0.0039870
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0038524, 0.0039716
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077896, 0.0075309
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0254954, 0.0247542
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0065898, 0.0068076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126286, upper bound: 0.0123091
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126286, upper bound: 0.0123091
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011706, 0.0011679
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036992, 0.0036865
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053201, 0.0053392
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039135, 0.0039278
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0038983, 0.0039125
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076614, 0.0076304
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0251281, 0.0250393
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066736, 0.0066997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139806, upper bound: 0.0139073
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140932, upper bound: 0.0138958
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011803, 0.0011643
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037443, 0.0036689
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0052938, 0.0054067
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0038937, 0.0039787
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0038785, 0.0039633
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077715, 0.0075875
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0254435, 0.0249163
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066374, 0.0067924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126153, upper bound: 0.0121080
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126153, upper bound: 0.0121080
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011697, 0.0011730
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0036946, 0.0037096
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053549, 0.0053324
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039397, 0.0039227
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039243, 0.0039074
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0076503, 0.0076870
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0250963, 0.0252014
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067212, 0.0066903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0137848, upper bound: 0.0135546
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0137395, upper bound: 0.0135751
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011878, 0.0011740
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037457, 0.0036857
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053222, 0.0054133
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039147, 0.0039838
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0038993, 0.0039683
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078037, 0.0076475
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0254726, 0.0250492
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066739, 0.0068025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129986, upper bound: 0.0128256
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129986, upper bound: 0.0128256
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011878, 0.0011778
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037457, 0.0037037
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053491, 0.0054132
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039349, 0.0039838
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039195, 0.0039683
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078036, 0.0076914
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0254723, 0.0251748
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067108, 0.0068024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0135040, upper bound: 0.0131679
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0133637, upper bound: 0.0132798
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011925, 0.0011715
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037690, 0.0036754
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053067, 0.0054482
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039030, 0.0040101
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0038877, 0.0039946
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078607, 0.0076222
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0256358, 0.0249768
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066526, 0.0068504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141194, upper bound: 0.0137690
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139806, upper bound: 0.0139073
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011915, 0.0011766
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037645, 0.0036981
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053408, 0.0054414
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039287, 0.0040050
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039133, 0.0039895
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078496, 0.0076779
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0256040, 0.0251361
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066994, 0.0068411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0107919, upper bound: 0.0106982
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0107919, upper bound: 0.0106982
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011919, 0.0011640
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037958, 0.0036685
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0052965, 0.0054884
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0038963, 0.0040402
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0038811, 0.0040248
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078860, 0.0075696
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0258234, 0.0249283
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066209, 0.0068869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0132379, upper bound: 0.0125443
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130757, upper bound: 0.0126739
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011890, 0.0011719
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037822, 0.0037098
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053584, 0.0054680
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039428, 0.0040249
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039275, 0.0040095
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078528, 0.0076704
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0257282, 0.0252171
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067058, 0.0068589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0134939, upper bound: 0.0129699
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0134939, upper bound: 0.0129743
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011754, 0.0011619
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037182, 0.0036587
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0052819, 0.0053728
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0038854, 0.0039536
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0038703, 0.0039384
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077084, 0.0075565
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0252838, 0.0248615
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066110, 0.0067363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0135598, upper bound: 0.0132578
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0135598, upper bound: 0.0132578
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011797, 0.0011603
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037384, 0.0036513
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0052708, 0.0054030
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0038770, 0.0039764
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0038619, 0.0039611
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077577, 0.0075385
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0254251, 0.0248097
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0065958, 0.0067778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119059, upper bound: 0.0116281
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119059, upper bound: 0.0116281
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011866, 0.0011701
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037717, 0.0036949
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053371, 0.0054521
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039265, 0.0040130
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039112, 0.0039975
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078262, 0.0076388
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0256542, 0.0251173
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066794, 0.0068372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146379, upper bound: 0.0142419
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145058, upper bound: 0.0143289
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011805, 0.0011751
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037435, 0.0037183
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0053721, 0.0054099
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0039529, 0.0039813
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0039375, 0.0039658
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077574, 0.0076959
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0254572, 0.0252810
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0067275, 0.0067793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142538, upper bound: 0.0140144
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142538, upper bound: 0.0140144
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011867, 0.0011717
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037371, 0.0036676
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0052968, 0.0054004
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0038960, 0.0039742
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0038807, 0.0039587
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0077829, 0.0076114
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0254128, 0.0249277
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066398, 0.0067849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145974, upper bound: 0.0143445
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145937, upper bound: 0.0143558
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0005653, 0.0008282, -0.0005653, 0.0008282, -0.0011910, 0.0011705
1: -0.0013260, 0.0025909, -0.0013260, 0.0025909, -0.0037572, 0.0036619
2: 0.0124598, 0.0183258, 0.0124598, 0.0183258, -0.0052882, 0.0054305
3: -0.0012577, 0.0031533, -0.0012577, 0.0031533, -0.0038895, 0.0039968
4: -0.0055397, -0.0014710, -0.0055397, -0.0014710, -0.0040687, 0.0040687
5: 0.0066828, 0.0110857, 0.0066828, 0.0110857, -0.0038742, 0.0039813
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024620, 0.0024620
7: -0.0224653, -0.0129071, -0.0224653, -0.0129071, -0.0078318, 0.0075973
8: 0.9594252, 0.9868104, 0.9594252, 0.9868104, -0.0255531, 0.0248876
9: 0.0015295, 0.0095781, 0.0015295, 0.0095781, -0.0066280, 0.0068261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 2.93 + 598.32 = 601.25 seconds
