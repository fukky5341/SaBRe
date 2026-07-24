## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 21.8342855583


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167)
1: (-14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249)
2: (-17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199)
3: (-20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737)
4: (-18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998)
5: (-15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762)
6: (-14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840)
7: (-18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370)
8: (-19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403)
9: (-14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.91 + 5.89 = 7.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -21.8561417, upper bound: 21.8561417

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8510480, upper bound: 21.8510480
time: 15.79 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8510480, upper bound: 21.8510480
time: 14.54 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 30.57 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 30.57
Output dim: 7, lower bound: -21.8510480, upper bound: 21.8510480
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 30.57
Output dim: 7, lower bound: -21.8510480, upper bound: 21.8510480

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8505848, upper bound: 21.8505848
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8505848, upper bound: 21.8505848
time: 3.24 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8505848, upper bound: 21.8505848
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8505848, upper bound: 21.8505848
time: 3.22 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 8.91 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 8.91
Output dim: 7, lower bound: -21.8505848, upper bound: 21.8505848
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 8.91
Output dim: 7, lower bound: -21.8505848, upper bound: 21.8505848
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 8.91
Output dim: 7, lower bound: -21.8505848, upper bound: 21.8505848
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 8.91
Output dim: 7, lower bound: -21.8505848, upper bound: 21.8505848

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8492672, upper bound: 21.8492599
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8492599, upper bound: 21.8492672
time: 5.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8492672, upper bound: 21.8492599
time: 6.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8492599, upper bound: 21.8492672
time: 3.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8492672, upper bound: 21.8492599
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8492599, upper bound: 21.8492672
time: 5.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 2.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8492672, upper bound: 21.8492599
time: 3.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8492599, upper bound: 21.8492672
time: 4.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 10.54 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.54
Output dim: 7, lower bound: -21.8492672, upper bound: 21.8492599
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.54
Output dim: 7, lower bound: -21.8492599, upper bound: 21.8492672
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.54
Output dim: 7, lower bound: -21.8492672, upper bound: 21.8492599
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.54
Output dim: 7, lower bound: -21.8492599, upper bound: 21.8492672
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.54
Output dim: 7, lower bound: -21.8492672, upper bound: 21.8492599
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.54
Output dim: 7, lower bound: -21.8492599, upper bound: 21.8492672
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.54
Output dim: 7, lower bound: -21.8492672, upper bound: 21.8492599
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.54
Output dim: 7, lower bound: -21.8492599, upper bound: 21.8492672

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8469984, upper bound: 21.8469912
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8469985, upper bound: 21.8469912
time: 4.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8469912, upper bound: 21.8469985
time: 5.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8469912, upper bound: 21.8469985
time: 4.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8469984, upper bound: 21.8469912
time: 4.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8469985, upper bound: 21.8469912
time: 4.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 2.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8469912, upper bound: 21.8469985
time: 6.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8469912, upper bound: 21.8469984
time: 3.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8469984, upper bound: 21.8469912
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8469985, upper bound: 21.8469912
time: 3.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8469912, upper bound: 21.8469985
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8469912, upper bound: 21.8469985
time: 3.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8469984, upper bound: 21.8469912
time: 5.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8469985, upper bound: 21.8469912
time: 3.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8469912, upper bound: 21.8469985
time: 4.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8469912, upper bound: 21.8469984
time: 3.82 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 10.59 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 7, lower bound: -21.8469984, upper bound: 21.8469912
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 7, lower bound: -21.8469985, upper bound: 21.8469912
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 7, lower bound: -21.8469912, upper bound: 21.8469985
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 7, lower bound: -21.8469912, upper bound: 21.8469985
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 7, lower bound: -21.8469984, upper bound: 21.8469912
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 7, lower bound: -21.8469985, upper bound: 21.8469912
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 7, lower bound: -21.8469912, upper bound: 21.8469985
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 7, lower bound: -21.8469912, upper bound: 21.8469984
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 7, lower bound: -21.8469984, upper bound: 21.8469912
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 7, lower bound: -21.8469985, upper bound: 21.8469912
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 7, lower bound: -21.8469912, upper bound: 21.8469985
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 7, lower bound: -21.8469912, upper bound: 21.8469985
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 7, lower bound: -21.8469984, upper bound: 21.8469912
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 7, lower bound: -21.8469985, upper bound: 21.8469912
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 7, lower bound: -21.8469912, upper bound: 21.8469985
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 7, lower bound: -21.8469912, upper bound: 21.8469984

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
time: 4.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
time: 6.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
time: 6.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
time: 6.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
time: 10.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
time: 6.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
time: 4.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
time: 88.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
time: 4.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
time: 3.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
time: 6.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
time: 6.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 2.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
time: 6.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
time: 11.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
time: 3.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
time: 6.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
time: 5.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
time: 6.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
time: 5.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
time: 8.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
time: 6.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
time: 17.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
time: 4.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
time: 5.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
time: 3.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
time: 4.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
time: 5.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
time: 6.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
time: 6.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
time: 24.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
time: 3.68 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 30.42 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.42
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8341836, upper bound: 21.8341691
time: 6.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8341836, upper bound: 21.8341691
time: 6.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8341836, upper bound: 21.8341691
time: 5.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8341836, upper bound: 21.8341691
time: 5.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8341784, upper bound: 21.8341737
time: 19.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8341784, upper bound: 21.8341737
time: 19.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -16.9740028, 13.6893234, -16.9740028, 13.6893234, -30.6633148, 30.6633167
1: -14.0589867, 11.9158411, -14.0589867, 11.9158411, -25.9748268, 25.9748249
2: -17.3404541, 10.5976658, -17.3404541, 10.5976658, -27.9381199, 27.9381199
3: -20.5597763, 9.2333050, -20.5597763, 9.2333050, -29.7930794, 29.7930737
4: -18.3966045, 14.3898029, -18.3966045, 14.3898029, -32.7863998, 32.7863998
5: -15.9375210, 12.1460552, -15.9375210, 12.1460552, -28.0835762, 28.0835762
6: -14.7187529, 15.9408302, -14.7187529, 15.9408302, -30.6595840, 30.6595840
7: -18.2727642, 10.8036785, -18.2727642, 10.8036785, -29.0764408, 29.0764370
8: -19.0032673, 13.1205730, -19.0032673, 13.1205730, -32.1238403, 32.1238403
9: -14.8070126, 15.1359177, -14.8070126, 15.1359177, -29.9429302, 29.9429302

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8341784, upper bound: 21.8341737
time: 30.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8341784, upper bound: 21.8341737
time: 3.95 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 36.85 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 36.85
Output dim: 7, lower bound: -21.8341836, upper bound: 21.8341691
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 36.85
Output dim: 7, lower bound: -21.8341836, upper bound: 21.8341691
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 36.85
Output dim: 7, lower bound: -21.8341836, upper bound: 21.8341691
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 36.85
Output dim: 7, lower bound: -21.8341836, upper bound: 21.8341691
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 36.85
Output dim: 7, lower bound: -21.8341784, upper bound: 21.8341737
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 36.85
Output dim: 7, lower bound: -21.8341784, upper bound: 21.8341737
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 36.85
Output dim: 7, lower bound: -21.8341784, upper bound: 21.8341737
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 36.85
Output dim: 7, lower bound: -21.8341784, upper bound: 21.8341737
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468586, upper bound: 21.8468570
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468587, upper bound: 21.8468569
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468569, upper bound: 21.8468587
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.85
Output dim: 7, lower bound: -21.8468570, upper bound: 21.8468586

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 7.79 + 618.72 = 626.51 seconds
