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
execution time: IAR + RelationalAnalysis = 2.14 + 12.67 = 14.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -23.7325583, upper bound: 23.7325576

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7275859, upper bound: 23.7275859
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7275859, upper bound: 23.7275859
time: 4.10 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.76 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.76
Output dim: 1, lower bound: -23.7275859, upper bound: 23.7275859
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.76
Output dim: 1, lower bound: -23.7275859, upper bound: 23.7275859

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

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7273930, upper bound: 23.7273950
time: 4.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7273950, upper bound: 23.7273930
time: 11.55 seconds

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

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7273930, upper bound: 23.7273950
time: 4.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7273950, upper bound: 23.7273930
time: 11.82 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 18.65 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 18.65
Output dim: 1, lower bound: -23.7273930, upper bound: 23.7273950
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 18.65
Output dim: 1, lower bound: -23.7273950, upper bound: 23.7273930
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 18.65
Output dim: 1, lower bound: -23.7273930, upper bound: 23.7273950
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 18.65
Output dim: 1, lower bound: -23.7273950, upper bound: 23.7273930

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

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254612, upper bound: 23.7254659
time: 6.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254612, upper bound: 23.7254659
time: 6.15 seconds

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

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254658, upper bound: 23.7254612
time: 18.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254659, upper bound: 23.7254612
time: 9.62 seconds

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

Time for backsubstitution: 2.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254612, upper bound: 23.7254658
time: 6.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254612, upper bound: 23.7254659
time: 6.81 seconds

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

Time for backsubstitution: 2.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254658, upper bound: 23.7254612
time: 6.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254658, upper bound: 23.7254612
time: 10.62 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 22.16 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 22.16
Output dim: 1, lower bound: -23.7254612, upper bound: 23.7254659
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 22.16
Output dim: 1, lower bound: -23.7254612, upper bound: 23.7254659
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 22.16
Output dim: 1, lower bound: -23.7254658, upper bound: 23.7254612
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 22.16
Output dim: 1, lower bound: -23.7254659, upper bound: 23.7254612
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 22.16
Output dim: 1, lower bound: -23.7254612, upper bound: 23.7254658
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 22.16
Output dim: 1, lower bound: -23.7254612, upper bound: 23.7254659
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 22.16
Output dim: 1, lower bound: -23.7254658, upper bound: 23.7254612
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 22.16
Output dim: 1, lower bound: -23.7254658, upper bound: 23.7254612

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

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254612, upper bound: 23.7254657
time: 6.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254611, upper bound: 23.7254659
time: 7.50 seconds

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

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254612, upper bound: 23.7254657
time: 7.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254611, upper bound: 23.7254659
time: 5.96 seconds

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

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254659, upper bound: 23.7254611
time: 6.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254657, upper bound: 23.7254612
time: 4.87 seconds

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

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254659, upper bound: 23.7254611
time: 12.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254657, upper bound: 23.7254612
time: 13.99 seconds

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

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254612, upper bound: 23.7254657
time: 6.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254611, upper bound: 23.7254659
time: 8.74 seconds

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

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254612, upper bound: 23.7254657
time: 4.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254611, upper bound: 23.7254659
time: 5.51 seconds

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

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254659, upper bound: 23.7254611
time: 6.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254657, upper bound: 23.7254612
time: 9.47 seconds

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

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254659, upper bound: 23.7254611
time: 9.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254657, upper bound: 23.7254612
time: 10.24 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 22.07 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 22.07
Output dim: 1, lower bound: -23.7254612, upper bound: 23.7254657
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 22.07
Output dim: 1, lower bound: -23.7254611, upper bound: 23.7254659
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 22.07
Output dim: 1, lower bound: -23.7254612, upper bound: 23.7254657
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 22.07
Output dim: 1, lower bound: -23.7254611, upper bound: 23.7254659
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 22.07
Output dim: 1, lower bound: -23.7254659, upper bound: 23.7254611
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 22.07
Output dim: 1, lower bound: -23.7254657, upper bound: 23.7254612
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 22.07
Output dim: 1, lower bound: -23.7254659, upper bound: 23.7254611
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 22.07
Output dim: 1, lower bound: -23.7254657, upper bound: 23.7254612
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 22.07
Output dim: 1, lower bound: -23.7254612, upper bound: 23.7254657
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 22.07
Output dim: 1, lower bound: -23.7254611, upper bound: 23.7254659
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 22.07
Output dim: 1, lower bound: -23.7254612, upper bound: 23.7254657
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 22.07
Output dim: 1, lower bound: -23.7254611, upper bound: 23.7254659
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 22.07
Output dim: 1, lower bound: -23.7254659, upper bound: 23.7254611
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 22.07
Output dim: 1, lower bound: -23.7254657, upper bound: 23.7254612
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 22.07
Output dim: 1, lower bound: -23.7254659, upper bound: 23.7254611
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 22.07
Output dim: 1, lower bound: -23.7254657, upper bound: 23.7254612

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

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198100
time: 6.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198100
time: 4.93 seconds

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

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
time: 4.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
time: 4.67 seconds

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

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198100
time: 5.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198100
time: 5.35 seconds

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

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
time: 7.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
time: 8.78 seconds

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

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198093
time: 5.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198094
time: 6.80 seconds

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

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099
time: 5.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099
time: 5.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198093
time: 5.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198094
time: 6.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198098
time: 6.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198100, upper bound: 23.7198099
time: 8.82 seconds

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

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198100
time: 7.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198099
time: 7.58 seconds

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

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
time: 8.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
time: 7.63 seconds

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

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198099
time: 5.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198100
time: 4.59 seconds

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

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
time: 7.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
time: 8.66 seconds

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

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198093
time: 5.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198094
time: 5.45 seconds

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

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099
time: 5.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099
time: 5.00 seconds

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

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198093
time: 5.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198093
time: 5.49 seconds

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

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099
time: 10.51 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 18.05 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198100
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198100
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198100
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198100
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198093
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198094
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198093
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198094
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198098
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198100, upper bound: 23.7198099
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198100
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198099
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198099
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198100
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198093
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198094
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198093
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198093
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.05
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099

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

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7192153, upper bound: 23.7192159
time: 25.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7192162, upper bound: 23.7192149
time: 5.93 seconds

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

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7192153, upper bound: 23.7192159
time: 6.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7192162, upper bound: 23.7192149
time: 4.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7192143, upper bound: 23.7192166
time: 6.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7192150, upper bound: 23.7192158
time: 5.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7192143, upper bound: 23.7192166
time: 10.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7192150, upper bound: 23.7192157
time: 5.03 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 17.24 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 17.24
Output dim: 1, lower bound: -23.7192153, upper bound: 23.7192159
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 17.24
Output dim: 1, lower bound: -23.7192162, upper bound: 23.7192149
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 17.24
Output dim: 1, lower bound: -23.7192153, upper bound: 23.7192159
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 17.24
Output dim: 1, lower bound: -23.7192162, upper bound: 23.7192149
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 17.24
Output dim: 1, lower bound: -23.7192143, upper bound: 23.7192166
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 17.24
Output dim: 1, lower bound: -23.7192150, upper bound: 23.7192158
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 17.24
Output dim: 1, lower bound: -23.7192143, upper bound: 23.7192166
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 17.24
Output dim: 1, lower bound: -23.7192150, upper bound: 23.7192157
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198100
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198100
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198093
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198094
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198093
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198094
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198098
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198100, upper bound: 23.7198099
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198100
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198099
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198099
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198098, upper bound: 23.7198100
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198093, upper bound: 23.7198104
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198093
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198094
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198093
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198104, upper bound: 23.7198093
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.24
Output dim: 1, lower bound: -23.7198099, upper bound: 23.7198099

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 14.81 + 595.97 = 610.78 seconds
