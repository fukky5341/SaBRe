## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.7875192


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813)
1: (-0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893)
2: (-0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501)
3: (0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201)
4: (-0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563)
5: (-0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932)
6: (-0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614)
7: (-0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642)
8: (-0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244)
9: (-0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 2.80 = 4.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8197644, upper bound: 0.8197494
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8197494, upper bound: 0.8197644
time: 1.46 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.02 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.02
Output dim: 3, lower bound: -0.8197644, upper bound: 0.8197494
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.02
Output dim: 3, lower bound: -0.8197494, upper bound: 0.8197644

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8104811, upper bound: 0.8104706
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8104811, upper bound: 0.8104706
time: 1.40 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8104706, upper bound: 0.8104811
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8104706, upper bound: 0.8104811
time: 1.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.54 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 3, lower bound: -0.8104811, upper bound: 0.8104706
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 3, lower bound: -0.8104811, upper bound: 0.8104706
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 3, lower bound: -0.8104706, upper bound: 0.8104811
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 3, lower bound: -0.8104706, upper bound: 0.8104811

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937643, upper bound: 0.7937185
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937643, upper bound: 0.7937185
time: 1.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937643, upper bound: 0.7937185
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937643, upper bound: 0.7937185
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937185, upper bound: 0.7937643
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937185, upper bound: 0.7937643
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937185, upper bound: 0.7937643
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937185, upper bound: 0.7937643
time: 1.47 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.30 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.30
Output dim: 3, lower bound: -0.7937643, upper bound: 0.7937185
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.30
Output dim: 3, lower bound: -0.7937643, upper bound: 0.7937185
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.30
Output dim: 3, lower bound: -0.7937643, upper bound: 0.7937185
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.30
Output dim: 3, lower bound: -0.7937643, upper bound: 0.7937185
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.30
Output dim: 3, lower bound: -0.7937185, upper bound: 0.7937643
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.30
Output dim: 3, lower bound: -0.7937185, upper bound: 0.7937643
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.30
Output dim: 3, lower bound: -0.7937185, upper bound: 0.7937643
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.30
Output dim: 3, lower bound: -0.7937185, upper bound: 0.7937643

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932306, upper bound: 0.7931820
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932213, upper bound: 0.7931830
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932306, upper bound: 0.7931819
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932213, upper bound: 0.7931830
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932306, upper bound: 0.7931820
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932213, upper bound: 0.7931830
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932306, upper bound: 0.7931820
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932213, upper bound: 0.7931830
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931830, upper bound: 0.7932213
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931819, upper bound: 0.7932306
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931830, upper bound: 0.7932213
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931819, upper bound: 0.7932306
time: 1.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931830, upper bound: 0.7932213
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931819, upper bound: 0.7932306
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931830, upper bound: 0.7932213
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931819, upper bound: 0.7932306
time: 1.48 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 3, lower bound: -0.7932306, upper bound: 0.7931820
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 3, lower bound: -0.7932213, upper bound: 0.7931830
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 3, lower bound: -0.7932306, upper bound: 0.7931819
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 3, lower bound: -0.7932213, upper bound: 0.7931830
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 3, lower bound: -0.7932306, upper bound: 0.7931820
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 3, lower bound: -0.7932213, upper bound: 0.7931830
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 3, lower bound: -0.7932306, upper bound: 0.7931820
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 3, lower bound: -0.7932213, upper bound: 0.7931830
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 3, lower bound: -0.7931830, upper bound: 0.7932213
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 3, lower bound: -0.7931819, upper bound: 0.7932306
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 3, lower bound: -0.7931830, upper bound: 0.7932213
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 3, lower bound: -0.7931819, upper bound: 0.7932306
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 3, lower bound: -0.7931830, upper bound: 0.7932213
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 3, lower bound: -0.7931819, upper bound: 0.7932306
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 3, lower bound: -0.7931830, upper bound: 0.7932213
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 6.03
Output dim: 3, lower bound: -0.7931819, upper bound: 0.7932306

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932218, upper bound: 0.7931516
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932116, upper bound: 0.7931593
time: 2.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932119, upper bound: 0.7931527
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932041, upper bound: 0.7931600
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932218, upper bound: 0.7931516
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932116, upper bound: 0.7931593
time: 2.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932119, upper bound: 0.7931527
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932041, upper bound: 0.7931600
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932218, upper bound: 0.7931516
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932116, upper bound: 0.7931593
time: 1.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932119, upper bound: 0.7931527
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932041, upper bound: 0.7931600
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932218, upper bound: 0.7931516
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932116, upper bound: 0.7931593
time: 1.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932119, upper bound: 0.7931527
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932041, upper bound: 0.7931600
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931600, upper bound: 0.7932041
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931527, upper bound: 0.7932119
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931593, upper bound: 0.7932116
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931516, upper bound: 0.7932218
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931600, upper bound: 0.7932041
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931527, upper bound: 0.7932119
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931593, upper bound: 0.7932116
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931516, upper bound: 0.7932218
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931600, upper bound: 0.7932041
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931527, upper bound: 0.7932119
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931593, upper bound: 0.7932116
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931516, upper bound: 0.7932218
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931600, upper bound: 0.7932041
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931527, upper bound: 0.7932119
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931593, upper bound: 0.7932116
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931516, upper bound: 0.7932218
time: 1.28 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7932218, upper bound: 0.7931516
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7932116, upper bound: 0.7931593
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7932119, upper bound: 0.7931527
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7932041, upper bound: 0.7931600
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7932218, upper bound: 0.7931516
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7932116, upper bound: 0.7931593
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7932119, upper bound: 0.7931527
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7932041, upper bound: 0.7931600
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7932218, upper bound: 0.7931516
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7932116, upper bound: 0.7931593
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7932119, upper bound: 0.7931527
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7932041, upper bound: 0.7931600
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7932218, upper bound: 0.7931516
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7932116, upper bound: 0.7931593
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7932119, upper bound: 0.7931527
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7932041, upper bound: 0.7931600
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7931600, upper bound: 0.7932041
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7931527, upper bound: 0.7932119
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7931593, upper bound: 0.7932116
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7931516, upper bound: 0.7932218
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7931600, upper bound: 0.7932041
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7931527, upper bound: 0.7932119
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7931593, upper bound: 0.7932116
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7931516, upper bound: 0.7932218
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7931600, upper bound: 0.7932041
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7931527, upper bound: 0.7932119
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7931593, upper bound: 0.7932116
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7931516, upper bound: 0.7932218
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7931600, upper bound: 0.7932041
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7931527, upper bound: 0.7932119
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7931593, upper bound: 0.7932116
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.43
Output dim: 3, lower bound: -0.7931516, upper bound: 0.7932218

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813
1: -0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893
2: -0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501
3: 0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201
4: -0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563
5: -0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932
6: -0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614
7: -0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642
8: -0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244
9: -0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.02 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.50
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 4.20 + 362.08 = 366.27 seconds
