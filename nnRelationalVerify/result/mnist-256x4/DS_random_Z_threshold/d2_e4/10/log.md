## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 23.708826041400002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808)
1: (-18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632)
2: (-22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983)
3: (-25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631)
4: (-23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213)
5: (-18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036)
6: (-19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349)
7: (-23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605)
8: (-27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121)
9: (-16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.84 + 12.47 = 13.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -23.7325583, upper bound: 23.7325576

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7266307, upper bound: 23.7266307
time: 15.56 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7266307, upper bound: 23.7266307
time: 9.03 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 24.60 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 24.60
Output dim: 1, lower bound: -23.7266307, upper bound: 23.7266307
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 24.60
Output dim: 1, lower bound: -23.7266307, upper bound: 23.7266307

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7257668, upper bound: 23.7257668
time: 6.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7257668, upper bound: 23.7257668
time: 4.93 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7252397, upper bound: 23.7252396
time: 4.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7252397, upper bound: 23.7252397
time: 4.92 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 10.68 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 10.68
Output dim: 1, lower bound: -23.7257668, upper bound: 23.7257668
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 10.68
Output dim: 1, lower bound: -23.7257668, upper bound: 23.7257668
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 10.68
Output dim: 1, lower bound: -23.7252397, upper bound: 23.7252396
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 10.68
Output dim: 1, lower bound: -23.7252397, upper bound: 23.7252397

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7234759, upper bound: 23.7234859
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7234759, upper bound: 23.7234859
time: 3.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7158453, upper bound: 23.7158445
time: 6.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7158453, upper bound: 23.7158445
time: 6.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7252390, upper bound: 23.7252390
time: 5.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7252390, upper bound: 23.7252390
time: 5.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7248980, upper bound: 23.7248980
time: 4.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7248980, upper bound: 23.7248980
time: 4.62 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 10.01 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.01
Output dim: 1, lower bound: -23.7234759, upper bound: 23.7234859
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.01
Output dim: 1, lower bound: -23.7234759, upper bound: 23.7234859
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.01
Output dim: 1, lower bound: -23.7158453, upper bound: 23.7158445
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.01
Output dim: 1, lower bound: -23.7158453, upper bound: 23.7158445
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.01
Output dim: 1, lower bound: -23.7252390, upper bound: 23.7252390
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.01
Output dim: 1, lower bound: -23.7252390, upper bound: 23.7252390
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.01
Output dim: 1, lower bound: -23.7248980, upper bound: 23.7248980
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.01
Output dim: 1, lower bound: -23.7248980, upper bound: 23.7248980

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7210459, upper bound: 23.7210515
time: 19.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7210459, upper bound: 23.7210516
time: 7.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7193178, upper bound: 23.7193239
time: 6.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7193178, upper bound: 23.7193240
time: 10.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7093335, upper bound: 23.7093334
time: 4.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7093335, upper bound: 23.7093334
time: 4.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -23.7081861, upper bound: 23.7081852
time: 4.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -23.7081861, upper bound: 23.7081853
time: 5.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7247459, upper bound: 23.7247459
time: 6.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7247459, upper bound: 23.7247459
time: 11.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 118

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7160447, upper bound: 23.7160421
time: 7.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7160447, upper bound: 23.7160421
time: 7.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7216599, upper bound: 23.7216599
time: 5.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7216599, upper bound: 23.7216599
time: 4.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7216599, upper bound: 23.7216598
time: 6.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7216599, upper bound: 23.7216598
time: 4.95 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 12.42 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.42
Output dim: 1, lower bound: -23.7210459, upper bound: 23.7210515
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.42
Output dim: 1, lower bound: -23.7210459, upper bound: 23.7210516
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.42
Output dim: 1, lower bound: -23.7193178, upper bound: 23.7193239
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.42
Output dim: 1, lower bound: -23.7193178, upper bound: 23.7193240
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.42
Output dim: 1, lower bound: -23.7093335, upper bound: 23.7093334
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.42
Output dim: 1, lower bound: -23.7093335, upper bound: 23.7093334
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 12.42
Output dim: 1, lower bound: -23.7081861, upper bound: 23.7081852
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 12.42
Output dim: 1, lower bound: -23.7081861, upper bound: 23.7081853
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.42
Output dim: 1, lower bound: -23.7247459, upper bound: 23.7247459
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.42
Output dim: 1, lower bound: -23.7247459, upper bound: 23.7247459
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.42
Output dim: 1, lower bound: -23.7160447, upper bound: 23.7160421
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.42
Output dim: 1, lower bound: -23.7160447, upper bound: 23.7160421
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.42
Output dim: 1, lower bound: -23.7216599, upper bound: 23.7216599
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.42
Output dim: 1, lower bound: -23.7216599, upper bound: 23.7216599
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.42
Output dim: 1, lower bound: -23.7216599, upper bound: 23.7216598
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.42
Output dim: 1, lower bound: -23.7216599, upper bound: 23.7216598

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7192033, upper bound: 23.7192086
time: 7.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7192033, upper bound: 23.7192087
time: 8.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -23.7034791, upper bound: 23.7034820
time: 4.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -23.7034791, upper bound: 23.7034826
time: 4.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 59

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -23.7027436, upper bound: 23.7027457
time: 4.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -23.7027436, upper bound: 23.7027457
time: 5.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7142042, upper bound: 23.7142060
time: 4.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7142042, upper bound: 23.7142060
time: 5.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7093335, upper bound: 23.7093312
time: 6.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7093319, upper bound: 23.7093334
time: 3.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7093335, upper bound: 23.7093331
time: 8.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7093334, upper bound: 23.7093334
time: 5.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7187473, upper bound: 23.7187519
time: 6.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7187473, upper bound: 23.7187519
time: 6.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7235998, upper bound: 23.7235998
time: 4.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7235998, upper bound: 23.7235998
time: 5.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -23.7082399, upper bound: 23.7082355
time: 6.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -23.7082399, upper bound: 23.7082355
time: 7.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7114567, upper bound: 23.7114567
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7114567, upper bound: 23.7114567
time: 5.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 169

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7210682, upper bound: 23.7210684
time: 4.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7210682, upper bound: 23.7210684
time: 4.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7213154, upper bound: 23.7213010
time: 7.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7213009, upper bound: 23.7213158
time: 5.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 169

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7204682, upper bound: 23.7204677
time: 12.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7204682, upper bound: 23.7204676
time: 8.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7204849, upper bound: 23.7204885
time: 4.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7204885, upper bound: 23.7204848
time: 6.61 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 12.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7192033, upper bound: 23.7192086
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7192033, upper bound: 23.7192087
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7034791, upper bound: 23.7034820
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7034791, upper bound: 23.7034826
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7027436, upper bound: 23.7027457
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7027436, upper bound: 23.7027457
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7142042, upper bound: 23.7142060
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7142042, upper bound: 23.7142060
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7093335, upper bound: 23.7093312
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7093319, upper bound: 23.7093334
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7093335, upper bound: 23.7093331
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7093334, upper bound: 23.7093334
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7187473, upper bound: 23.7187519
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7187473, upper bound: 23.7187519
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7235998, upper bound: 23.7235998
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7235998, upper bound: 23.7235998
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7082399, upper bound: 23.7082355
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7082399, upper bound: 23.7082355
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7114567, upper bound: 23.7114567
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7114567, upper bound: 23.7114567
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7210682, upper bound: 23.7210684
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7210682, upper bound: 23.7210684
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7213154, upper bound: 23.7213010
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7213009, upper bound: 23.7213158
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7204682, upper bound: 23.7204677
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7204682, upper bound: 23.7204676
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7204849, upper bound: 23.7204885
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.22
Output dim: 1, lower bound: -23.7204885, upper bound: 23.7204848

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 203

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 112

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7191993, upper bound: 23.7192087
time: 8.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7192034, upper bound: 23.7192047
time: 23.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 59

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7098462, upper bound: 23.7098529
time: 5.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7098462, upper bound: 23.7098529
time: 4.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 67

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7142042, upper bound: 23.7142052
time: 5.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7142034, upper bound: 23.7142060
time: 5.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7105017, upper bound: 23.7105047
time: 6.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7105017, upper bound: 23.7105047
time: 5.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7090280, upper bound: 23.7090244
time: 6.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7090274, upper bound: 23.7090257
time: 5.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7093316, upper bound: 23.7093334
time: 5.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7093319, upper bound: 23.7093334
time: 3.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7093293, upper bound: 23.7093331
time: 5.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7093335, upper bound: 23.7093292
time: 22.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -23.7088111, upper bound: 23.7088109
time: 6.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -23.7088115, upper bound: 23.7088109
time: 25.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -18.7413445, 16.8435440, -18.7413445, 16.8435440, -35.5848808, 35.5848808
1: -18.2360153, 11.6385536, -18.2360153, 11.6385536, -29.8745670, 29.8745632
2: -22.0091190, 14.2606821, -22.0091190, 14.2606821, -36.2697983, 36.2697983
3: -25.8568459, 12.4839172, -25.8568459, 12.4839172, -38.3407631, 38.3407631
4: -23.7274017, 15.2047176, -23.7274017, 15.2047176, -38.9321213, 38.9321213
5: -18.3004799, 16.2541256, -18.3004799, 16.2541256, -34.5546036, 34.5546036
6: -19.1975403, 17.5571003, -19.1975403, 17.5571003, -36.7546387, 36.7546349
7: -23.2292156, 16.8357449, -23.2292156, 16.8357449, -40.0649605, 40.0649605
8: -27.6445599, 14.1375542, -27.6445599, 14.1375542, -41.7821121, 41.7821121
9: -16.9823532, 19.0270061, -16.9823532, 19.0270061, -36.0093613, 36.0093613

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7187158, upper bound: 23.7187185
time: 4.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7187108, upper bound: 23.7187203
time: 4.93 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 10.49 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7191993, upper bound: 23.7192087
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7192034, upper bound: 23.7192047
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7098462, upper bound: 23.7098529
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7098462, upper bound: 23.7098529
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7142042, upper bound: 23.7142052
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7142034, upper bound: 23.7142060
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7105017, upper bound: 23.7105047
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7105017, upper bound: 23.7105047
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7090280, upper bound: 23.7090244
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7090274, upper bound: 23.7090257
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7093316, upper bound: 23.7093334
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7093319, upper bound: 23.7093334
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7093293, upper bound: 23.7093331
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7093335, upper bound: 23.7093292
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7088111, upper bound: 23.7088109
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7088115, upper bound: 23.7088109
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7187158, upper bound: 23.7187185
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.49
Output dim: 1, lower bound: -23.7187108, upper bound: 23.7187203
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.49
Output dim: 1, lower bound: -23.7187473, upper bound: 23.7187519
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.49
Output dim: 1, lower bound: -23.7235998, upper bound: 23.7235998
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.49
Output dim: 1, lower bound: -23.7235998, upper bound: 23.7235998
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.49
Output dim: 1, lower bound: -23.7114567, upper bound: 23.7114567
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.49
Output dim: 1, lower bound: -23.7114567, upper bound: 23.7114567
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.49
Output dim: 1, lower bound: -23.7210682, upper bound: 23.7210684
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.49
Output dim: 1, lower bound: -23.7210682, upper bound: 23.7210684
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.49
Output dim: 1, lower bound: -23.7213154, upper bound: 23.7213010
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.49
Output dim: 1, lower bound: -23.7213009, upper bound: 23.7213158
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.49
Output dim: 1, lower bound: -23.7204682, upper bound: 23.7204677
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.49
Output dim: 1, lower bound: -23.7204682, upper bound: 23.7204676
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.49
Output dim: 1, lower bound: -23.7204849, upper bound: 23.7204885
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.49
Output dim: 1, lower bound: -23.7204885, upper bound: 23.7204848

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 13.31 + 588.32 = 601.63 seconds
