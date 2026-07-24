## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.314959394


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219)
1: (-0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226)
2: (-0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511)
3: (-0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240)
4: (-0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287)
5: (-0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725)
6: (-0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234)
7: (-0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564)
8: (-0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595)
9: (-0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.91 + 2.70 = 4.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 1.90 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.51 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.51
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.51
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 4.57 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 1.59 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.84 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.84
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.84
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.84
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.84
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 1.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 4.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
time: 1.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 7.82 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.82
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.82
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.82
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.82
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.82
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.82
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.82
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.82
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.37 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.04 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 2.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
time: 1.31 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 1, lower bound: -1.3251852, upper bound: 1.3251852

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 2.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 3.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
time: 1.26 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.51 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.51
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 2.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219
1: -0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226
2: -0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511
3: -0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240
4: -0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287
5: -0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725
6: -0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234
7: -0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564
8: -0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595
9: -0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
time: 1.52 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.67 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.67
Output dim: 1, lower bound: -1.3156233, upper bound: 1.3156233
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.67
Output dim: 1, lower bound: -1.3247253, upper bound: 1.3247253

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.62 + 597.43 = 602.05 seconds
