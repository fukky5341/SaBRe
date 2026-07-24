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
execution time: IAR + RelationalAnalysis = 0.80 + 5.71 = 6.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -21.8561417, upper bound: 21.8561417

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8460212, upper bound: 21.8460212
time: 35.72 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8460212, upper bound: 21.8460212
time: 35.60 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 71.34 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 71.34
Output dim: 7, lower bound: -21.8460212, upper bound: 21.8460212
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 71.34
Output dim: 7, lower bound: -21.8460212, upper bound: 21.8460212

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8433652, upper bound: 21.8433652
time: 6.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8433652, upper bound: 21.8433652
time: 6.20 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8173842, upper bound: 21.8173842
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8173842, upper bound: 21.8173842
time: 2.85 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 8.72 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 8.72
Output dim: 7, lower bound: -21.8433652, upper bound: 21.8433652
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 8.72
Output dim: 7, lower bound: -21.8433652, upper bound: 21.8433652
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 8.72
Output dim: 7, lower bound: -21.8173842, upper bound: 21.8173842
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 8.72
Output dim: 7, lower bound: -21.8173842, upper bound: 21.8173842

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8433640, upper bound: 21.8433640
time: 4.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8433640, upper bound: 21.8433640
time: 3.83 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8433652, upper bound: 21.8433645
time: 7.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8433645, upper bound: 21.8433652
time: 7.80 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 16.44 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.44
Output dim: 7, lower bound: -21.8433640, upper bound: 21.8433640
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.44
Output dim: 7, lower bound: -21.8433640, upper bound: 21.8433640
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.44
Output dim: 7, lower bound: -21.8433652, upper bound: 21.8433645
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.44
Output dim: 7, lower bound: -21.8433645, upper bound: 21.8433652

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8426684, upper bound: 21.8426684
time: 3.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8426684, upper bound: 21.8426684
time: 4.05 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 154

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8421331, upper bound: 21.8421331
time: 12.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8421331, upper bound: 21.8421331
time: 5.67 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8430704, upper bound: 21.8430698
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8430704, upper bound: 21.8430698
time: 3.71 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8377885, upper bound: 21.8377887
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8377885, upper bound: 21.8377887
time: 3.33 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 9.19 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.19
Output dim: 7, lower bound: -21.8426684, upper bound: 21.8426684
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.19
Output dim: 7, lower bound: -21.8426684, upper bound: 21.8426684
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.19
Output dim: 7, lower bound: -21.8421331, upper bound: 21.8421331
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.19
Output dim: 7, lower bound: -21.8421331, upper bound: 21.8421331
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.19
Output dim: 7, lower bound: -21.8430704, upper bound: 21.8430698
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.19
Output dim: 7, lower bound: -21.8430704, upper bound: 21.8430698
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.19
Output dim: 7, lower bound: -21.8377885, upper bound: 21.8377887
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.19
Output dim: 7, lower bound: -21.8377885, upper bound: 21.8377887

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8426684, upper bound: 21.8426673
time: 5.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8426673, upper bound: 21.8426684
time: 4.61 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8272157, upper bound: 21.8272190
time: 4.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8272157, upper bound: 21.8272190
time: 4.21 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8417540, upper bound: 21.8417540
time: 4.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8417540, upper bound: 21.8417540
time: 2.90 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 155

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8177748, upper bound: 21.8177615
time: 4.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8177748, upper bound: 21.8177615
time: 4.05 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8430702, upper bound: 21.8430698
time: 5.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8430704, upper bound: 21.8430698
time: 3.28 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8346512, upper bound: 21.8346484
time: 5.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8346512, upper bound: 21.8346484
time: 6.00 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8372104, upper bound: 21.8372123
time: 3.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8372104, upper bound: 21.8372123
time: 2.78 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 155

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8118993, upper bound: 21.8119004
time: 5.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8118993, upper bound: 21.8119002
time: 3.00 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 9.52 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.52
Output dim: 7, lower bound: -21.8426684, upper bound: 21.8426673
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.52
Output dim: 7, lower bound: -21.8426673, upper bound: 21.8426684
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 9.52
Output dim: 7, lower bound: -21.8272157, upper bound: 21.8272190
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 9.52
Output dim: 7, lower bound: -21.8272157, upper bound: 21.8272190
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.52
Output dim: 7, lower bound: -21.8417540, upper bound: 21.8417540
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.52
Output dim: 7, lower bound: -21.8417540, upper bound: 21.8417540
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 9.52
Output dim: 7, lower bound: -21.8177748, upper bound: 21.8177615
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 9.52
Output dim: 7, lower bound: -21.8177748, upper bound: 21.8177615
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.52
Output dim: 7, lower bound: -21.8430702, upper bound: 21.8430698
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.52
Output dim: 7, lower bound: -21.8430704, upper bound: 21.8430698
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.52
Output dim: 7, lower bound: -21.8346512, upper bound: 21.8346484
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.52
Output dim: 7, lower bound: -21.8346512, upper bound: 21.8346484
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.52
Output dim: 7, lower bound: -21.8372104, upper bound: 21.8372123
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.52
Output dim: 7, lower bound: -21.8372104, upper bound: 21.8372123
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 9.52
Output dim: 7, lower bound: -21.8118993, upper bound: 21.8119004
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 9.52
Output dim: 7, lower bound: -21.8118993, upper bound: 21.8119002

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8423389, upper bound: 21.8423334
time: 6.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8423363, upper bound: 21.8423377
time: 6.09 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8419723, upper bound: 21.8419724
time: 15.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8419723, upper bound: 21.8419724
time: 3.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8417540, upper bound: 21.8417537
time: 11.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8417537, upper bound: 21.8417540
time: 5.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8417540, upper bound: 21.8417537
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8417537, upper bound: 21.8417540
time: 27.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 126

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8398434, upper bound: 21.8398421
time: 4.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8398434, upper bound: 21.8398421
time: 6.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8425823, upper bound: 21.8425815
time: 5.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8425823, upper bound: 21.8425815
time: 3.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8272487, upper bound: 21.8272500
time: 8.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8272487, upper bound: 21.8272500
time: 7.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8338395, upper bound: 21.8338369
time: 5.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8338395, upper bound: 21.8338369
time: 4.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8366807, upper bound: 21.8366832
time: 16.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8366807, upper bound: 21.8366832
time: 2.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8370635, upper bound: 21.8370639
time: 4.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8370635, upper bound: 21.8370639
time: 6.16 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 11.81 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8423389, upper bound: 21.8423334
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8423363, upper bound: 21.8423377
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8419723, upper bound: 21.8419724
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8419723, upper bound: 21.8419724
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8417540, upper bound: 21.8417537
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8417537, upper bound: 21.8417540
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8417540, upper bound: 21.8417537
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8417537, upper bound: 21.8417540
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8398434, upper bound: 21.8398421
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8398434, upper bound: 21.8398421
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8425823, upper bound: 21.8425815
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8425823, upper bound: 21.8425815
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8272487, upper bound: 21.8272500
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8272487, upper bound: 21.8272500
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8338395, upper bound: 21.8338369
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8338395, upper bound: 21.8338369
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8366807, upper bound: 21.8366832
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8366807, upper bound: 21.8366832
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8370635, upper bound: 21.8370639
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.81
Output dim: 7, lower bound: -21.8370635, upper bound: 21.8370639

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8423385, upper bound: 21.8423334
time: 6.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8423389, upper bound: 21.8423332
time: 5.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8197438, upper bound: 21.8197512
time: 6.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -21.8197438, upper bound: 21.8197512
time: 3.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8361114, upper bound: 21.8361128
time: 5.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8361114, upper bound: 21.8361128
time: 4.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8401265, upper bound: 21.8401266
time: 6.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8401265, upper bound: 21.8401266
time: 3.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8362111, upper bound: 21.8362111
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8362111, upper bound: 21.8362111
time: 5.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 154

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8417537, upper bound: 21.8417505
time: 5.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8417500, upper bound: 21.8417540
time: 7.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8362111, upper bound: 21.8362111
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8362111, upper bound: 21.8362111
time: 3.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8417537, upper bound: 21.8417533
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8417532, upper bound: 21.8417540
time: 3.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8398434, upper bound: 21.8398412
time: 5.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8398416, upper bound: 21.8398421
time: 3.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8388124, upper bound: 21.8388083
time: 4.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8388124, upper bound: 21.8388083
time: 2.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8425823, upper bound: 21.8425809
time: 5.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8425815, upper bound: 21.8425815
time: 7.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8425823, upper bound: 21.8425709
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8425725, upper bound: 21.8425815
time: 46.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 154
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8343907, upper bound: 21.8343893
time: 6.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -21.8343888, upper bound: 21.8343912
time: 4.52 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 11.35 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8423385, upper bound: 21.8423334
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8423389, upper bound: 21.8423332
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8197438, upper bound: 21.8197512
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8197438, upper bound: 21.8197512
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8361114, upper bound: 21.8361128
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8361114, upper bound: 21.8361128
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8401265, upper bound: 21.8401266
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8401265, upper bound: 21.8401266
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8362111, upper bound: 21.8362111
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8362111, upper bound: 21.8362111
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8417537, upper bound: 21.8417505
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8417500, upper bound: 21.8417540
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8362111, upper bound: 21.8362111
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8362111, upper bound: 21.8362111
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8417537, upper bound: 21.8417533
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8417532, upper bound: 21.8417540
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8398434, upper bound: 21.8398412
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8398416, upper bound: 21.8398421
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8388124, upper bound: 21.8388083
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8388124, upper bound: 21.8388083
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8425823, upper bound: 21.8425809
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8425815, upper bound: 21.8425815
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8425823, upper bound: 21.8425709
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8425725, upper bound: 21.8425815
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8343907, upper bound: 21.8343893
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 11.35
Output dim: 7, lower bound: -21.8343888, upper bound: 21.8343912
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.35
Output dim: 7, lower bound: -21.8366807, upper bound: 21.8366832
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 11.35
Output dim: 7, lower bound: -21.8370635, upper bound: 21.8370639
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 11.35
Output dim: 7, lower bound: -21.8370635, upper bound: 21.8370639

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 6.51 + 603.02 = 609.53 seconds
