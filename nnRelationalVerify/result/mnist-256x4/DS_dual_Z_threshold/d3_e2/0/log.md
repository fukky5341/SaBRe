## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.7875191039999999


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.03 + 2.76 = 4.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8197644, upper bound: 0.8197494
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8197494, upper bound: 0.8197644
time: 1.39 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.05 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.05
Output dim: 3, lower bound: -0.8197644, upper bound: 0.8197494
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.05
Output dim: 3, lower bound: -0.8197494, upper bound: 0.8197644

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8104811, upper bound: 0.8104706
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8104811, upper bound: 0.8104706
time: 1.52 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8104706, upper bound: 0.8104811
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8104706, upper bound: 0.8104811
time: 1.64 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.59 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.59
Output dim: 3, lower bound: -0.8104811, upper bound: 0.8104706
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.59
Output dim: 3, lower bound: -0.8104811, upper bound: 0.8104706
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.59
Output dim: 3, lower bound: -0.8104706, upper bound: 0.8104811
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.59
Output dim: 3, lower bound: -0.8104706, upper bound: 0.8104811

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937643, upper bound: 0.7937185
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937643, upper bound: 0.7937185
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937643, upper bound: 0.7937185
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937643, upper bound: 0.7937185
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937185, upper bound: 0.7937643
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937185, upper bound: 0.7937643
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937185, upper bound: 0.7937643
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937185, upper bound: 0.7937643
time: 1.46 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.93 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.93
Output dim: 3, lower bound: -0.7937643, upper bound: 0.7937185
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.93
Output dim: 3, lower bound: -0.7937643, upper bound: 0.7937185
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.93
Output dim: 3, lower bound: -0.7937643, upper bound: 0.7937185
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.93
Output dim: 3, lower bound: -0.7937643, upper bound: 0.7937185
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.93
Output dim: 3, lower bound: -0.7937185, upper bound: 0.7937643
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.93
Output dim: 3, lower bound: -0.7937185, upper bound: 0.7937643
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.93
Output dim: 3, lower bound: -0.7937185, upper bound: 0.7937643
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.93
Output dim: 3, lower bound: -0.7937185, upper bound: 0.7937643

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932306, upper bound: 0.7931820
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932213, upper bound: 0.7931830
time: 1.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932306, upper bound: 0.7931819
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932213, upper bound: 0.7931830
time: 1.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932306, upper bound: 0.7931820
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932213, upper bound: 0.7931830
time: 1.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932306, upper bound: 0.7931820
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932213, upper bound: 0.7931830
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931830, upper bound: 0.7932213
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931819, upper bound: 0.7932306
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931830, upper bound: 0.7932213
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931819, upper bound: 0.7932306
time: 1.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931830, upper bound: 0.7932213
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931819, upper bound: 0.7932306
time: 1.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931830, upper bound: 0.7932213
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931819, upper bound: 0.7932306
time: 1.50 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 6.84 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 3, lower bound: -0.7932306, upper bound: 0.7931820
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 3, lower bound: -0.7932213, upper bound: 0.7931830
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 3, lower bound: -0.7932306, upper bound: 0.7931819
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 3, lower bound: -0.7932213, upper bound: 0.7931830
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 3, lower bound: -0.7932306, upper bound: 0.7931820
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 3, lower bound: -0.7932213, upper bound: 0.7931830
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 3, lower bound: -0.7932306, upper bound: 0.7931820
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 3, lower bound: -0.7932213, upper bound: 0.7931830
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 3, lower bound: -0.7931830, upper bound: 0.7932213
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 3, lower bound: -0.7931819, upper bound: 0.7932306
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 3, lower bound: -0.7931830, upper bound: 0.7932213
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 3, lower bound: -0.7931819, upper bound: 0.7932306
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 3, lower bound: -0.7931830, upper bound: 0.7932213
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 3, lower bound: -0.7931819, upper bound: 0.7932306
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 3, lower bound: -0.7931830, upper bound: 0.7932213
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.84
Output dim: 3, lower bound: -0.7931819, upper bound: 0.7932306

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932218, upper bound: 0.7931516
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932116, upper bound: 0.7931593
time: 2.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932119, upper bound: 0.7931527
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932041, upper bound: 0.7931600
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932218, upper bound: 0.7931516
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932116, upper bound: 0.7931593
time: 2.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932119, upper bound: 0.7931527
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932041, upper bound: 0.7931600
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932218, upper bound: 0.7931516
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932116, upper bound: 0.7931593
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932119, upper bound: 0.7931527
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932041, upper bound: 0.7931600
time: 1.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932218, upper bound: 0.7931516
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932116, upper bound: 0.7931593
time: 1.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932119, upper bound: 0.7931527
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932041, upper bound: 0.7931600
time: 1.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931600, upper bound: 0.7932041
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931527, upper bound: 0.7932119
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931593, upper bound: 0.7932116
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931516, upper bound: 0.7932218
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931600, upper bound: 0.7932041
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931527, upper bound: 0.7932119
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931593, upper bound: 0.7932116
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931516, upper bound: 0.7932218
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931600, upper bound: 0.7932041
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931527, upper bound: 0.7932119
time: 1.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931593, upper bound: 0.7932116
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931516, upper bound: 0.7932218
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931600, upper bound: 0.7932041
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931527, upper bound: 0.7932119
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931593, upper bound: 0.7932116
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7931516, upper bound: 0.7932218
time: 1.29 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.12 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7932218, upper bound: 0.7931516
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7932116, upper bound: 0.7931593
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7932119, upper bound: 0.7931527
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7932041, upper bound: 0.7931600
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7932218, upper bound: 0.7931516
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7932116, upper bound: 0.7931593
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7932119, upper bound: 0.7931527
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7932041, upper bound: 0.7931600
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7932218, upper bound: 0.7931516
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7932116, upper bound: 0.7931593
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7932119, upper bound: 0.7931527
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7932041, upper bound: 0.7931600
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7932218, upper bound: 0.7931516
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7932116, upper bound: 0.7931593
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7932119, upper bound: 0.7931527
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7932041, upper bound: 0.7931600
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7931600, upper bound: 0.7932041
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7931527, upper bound: 0.7932119
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7931593, upper bound: 0.7932116
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7931516, upper bound: 0.7932218
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7931600, upper bound: 0.7932041
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7931527, upper bound: 0.7932119
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7931593, upper bound: 0.7932116
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7931516, upper bound: 0.7932218
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7931600, upper bound: 0.7932041
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7931527, upper bound: 0.7932119
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7931593, upper bound: 0.7932116
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7931516, upper bound: 0.7932218
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7931600, upper bound: 0.7932041
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7931527, upper bound: 0.7932119
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7931593, upper bound: 0.7932116
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.12
Output dim: 3, lower bound: -0.7931516, upper bound: 0.7932218

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
time: 1.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
time: 1.03 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.18 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254080, upper bound: 0.7253972
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7254093, upper bound: 0.7253964
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253964, upper bound: 0.7254093
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 3, lower bound: -0.7253972, upper bound: 0.7254080

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.79 + 314.30 = 319.09 seconds
