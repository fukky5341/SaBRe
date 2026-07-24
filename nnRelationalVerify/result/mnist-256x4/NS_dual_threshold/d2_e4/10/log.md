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
execution time: IAR + RelationalAnalysis = 1.97 + 12.66 = 14.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -23.7325583, upper bound: 23.7325576

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7287462, upper bound: 23.7279236
time: 6.98 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7275859, upper bound: 23.7275859
time: 7.65 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 14.82 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 14.82
Output dim: 1, lower bound: -23.7287462, upper bound: 23.7279236
NS_A2, status: Status.UNKNOWN, split count: 1, time: 14.82
Output dim: 1, lower bound: -23.7275859, upper bound: 23.7275859

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -18.3535156, 16.5113792, -18.7413445, 16.8435440, -35.1970558, 35.2527199
1: -17.8811073, 11.3622904, -18.2360153, 11.6385536, -29.5196609, 29.5983047
2: -21.5553150, 13.9700518, -22.0091190, 14.2606821, -35.8159981, 35.9791641
3: -25.3515930, 12.2366562, -25.8568459, 12.4839172, -37.8355103, 38.0935020
4: -23.2681942, 14.8888998, -23.7274017, 15.2047176, -38.4729118, 38.6163025
5: -17.9131336, 15.9358969, -18.3004799, 16.2541256, -34.1672554, 34.2363739
6: -18.8194771, 17.1933479, -19.1975403, 17.5571003, -36.3765755, 36.3908844
7: -22.7875481, 16.4832325, -23.2292156, 16.8357449, -39.6232910, 39.7124481
8: -27.1288662, 13.8255348, -27.6445599, 14.1375542, -41.2664185, 41.4700928
9: -16.6105728, 18.6595650, -16.9823532, 19.0270061, -35.6375809, 35.6419067

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 64

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7275859, upper bound: 23.7275859
time: 8.83 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7275859, upper bound: 23.7275859
time: 5.22 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -21.2030487, 19.1075573, -18.2036858, 16.3854294, -37.5884743, 37.3112411
1: -20.5699940, 12.8583755, -17.7459068, 11.2595654, -31.8295555, 30.6042767
2: -24.9091492, 16.0702934, -21.3829422, 13.8602438, -38.7693939, 37.4532318
3: -29.5225773, 14.0788679, -25.1561470, 12.1428633, -41.6654396, 39.2350121
4: -26.9627037, 17.1359959, -23.0906143, 14.7661228, -41.7288208, 40.2266083
5: -20.6739845, 18.4434395, -17.7654152, 15.8133030, -36.4872894, 36.2088547
6: -21.8138485, 19.8117008, -18.6742210, 17.0571861, -38.8710327, 38.4859200
7: -26.4649277, 18.9766197, -22.6164608, 16.3488941, -42.8138161, 41.5930748
8: -31.5547485, 15.7893305, -26.9308643, 13.7076778, -45.2624245, 42.7201920
9: -19.0902290, 21.5709038, -16.4694710, 18.5172920, -37.6075134, 38.0403748

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7275859, upper bound: 23.7275859
time: 4.54 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7275859, upper bound: 23.7275859
time: 4.50 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 13.15 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 13.15
Output dim: 1, lower bound: -23.7275859, upper bound: 23.7275859
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 13.15
Output dim: 1, lower bound: -23.7275859, upper bound: 23.7275859
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 13.15
Output dim: 1, lower bound: -23.7275859, upper bound: 23.7275859
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 13.15
Output dim: 1, lower bound: -23.7275859, upper bound: 23.7275859

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -18.3535156, 16.5113792, -18.3535156, 16.5113792, -34.8648949, 34.8648949
1: -17.8811073, 11.3622904, -17.8811073, 11.3622904, -29.2433968, 29.2433968
2: -21.5553150, 13.9700518, -21.5553150, 13.9700518, -35.5253677, 35.5253677
3: -25.3515930, 12.2366562, -25.3515930, 12.2366562, -37.5882492, 37.5882492
4: -23.2681942, 14.8888998, -23.2681942, 14.8888998, -38.1570930, 38.1570930
5: -17.9131336, 15.9358969, -17.9131336, 15.9358969, -33.8490257, 33.8490295
6: -18.8194771, 17.1933479, -18.8194771, 17.1933479, -36.0128250, 36.0128250
7: -22.7875481, 16.4832325, -22.7875481, 16.4832325, -39.2707748, 39.2707748
8: -27.1288662, 13.8255348, -27.1288662, 13.8255348, -40.9543953, 40.9543991
9: -16.6105728, 18.6595650, -16.6105728, 18.6595650, -35.2701340, 35.2701340

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7144574, upper bound: 23.7143460
time: 5.46 seconds

## Relational analysis of NS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7232439, upper bound: 23.7228779
time: 5.82 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7287462, upper bound: 23.7279236
time: 7.88 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -18.3535156, 16.5113792, -21.2030487, 19.1075573, -37.4610748, 37.7144241
1: -17.8811073, 11.3622904, -20.5699940, 12.8583755, -30.7394810, 31.9322853
2: -21.5553150, 13.9700518, -24.9091492, 16.0702934, -37.6256104, 38.8792000
3: -25.3515930, 12.2366562, -29.5225773, 14.0788679, -39.4304619, 41.7592316
4: -23.2681942, 14.8888998, -26.9627037, 17.1359959, -40.4041901, 41.8516045
5: -17.9131336, 15.9358969, -20.6739845, 18.4434395, -36.3565712, 36.6098824
6: -18.8194771, 17.1933479, -21.8138485, 19.8117008, -38.6311798, 39.0071945
7: -22.7875481, 16.4832325, -26.4649277, 18.9766197, -41.7641678, 42.9481506
8: -27.1288662, 13.8255348, -31.5547485, 15.7893305, -42.9181976, 45.3802795
9: -16.6105728, 18.6595650, -19.0902290, 21.5709038, -38.1814728, 37.7497864

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 64

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7270140, upper bound: 23.7260778
time: 8.70 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7268970, upper bound: 23.7260453
time: 6.10 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -21.2030487, 19.1075573, -18.3498192, 16.5085697, -37.7116165, 37.4573746
1: -20.5699940, 12.8583755, -17.8775482, 11.3598633, -31.9298553, 30.7359200
2: -24.9091492, 16.0702934, -21.5512123, 13.9662323, -38.8753815, 37.6215057
3: -29.5225773, 14.0788679, -25.3469353, 12.2347212, -41.7572975, 39.4258041
4: -26.9627037, 17.1359959, -23.2639809, 14.8853245, -41.8480301, 40.3999786
5: -20.6739845, 18.4434395, -17.9097557, 15.9324074, -36.6063919, 36.3531952
6: -21.8138485, 19.8117008, -18.8156071, 17.1892796, -39.0031281, 38.6273041
7: -26.4649277, 18.9766197, -22.7826138, 16.4797325, -42.9446487, 41.7592316
8: -31.5547485, 15.7893305, -27.1238136, 13.8224106, -45.3771591, 42.9131432
9: -19.0902290, 21.5709038, -16.6065674, 18.6559601, -37.7461815, 38.1774635

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 64

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7257269, upper bound: 23.7259862
time: 5.03 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7256369, upper bound: 23.7256369
time: 5.35 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -21.2030487, 19.1075573, -21.2030487, 19.1075573, -40.3106079, 40.3106079
1: -20.5699940, 12.8583755, -20.5699940, 12.8583755, -33.4283676, 33.4283676
2: -24.9091492, 16.0702934, -24.9091492, 16.0702934, -40.9794426, 40.9794426
3: -29.5225773, 14.0788679, -29.5225773, 14.0788679, -43.6014442, 43.6014442
4: -26.9627037, 17.1359959, -26.9627037, 17.1359959, -44.0987015, 44.0987015
5: -20.6739845, 18.4434395, -20.6739845, 18.4434395, -39.1174240, 39.1174240
6: -21.8138485, 19.8117008, -21.8138485, 19.8117008, -41.6255493, 41.6255493
7: -26.4649277, 18.9766197, -26.4649277, 18.9766197, -45.4415436, 45.4415436
8: -31.5547485, 15.7893305, -31.5547485, 15.7893305, -47.3440781, 47.3440781
9: -19.0902290, 21.5709038, -19.0902290, 21.5709038, -40.6611214, 40.6611252

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7257269, upper bound: 23.7259863
time: 5.85 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7256369, upper bound: 23.7256369
time: 5.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 21.66 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 1, lower bound: -23.7232439, upper bound: 23.7228779
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 1, lower bound: -23.7287462, upper bound: 23.7279236
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 1, lower bound: -23.7270140, upper bound: 23.7260778
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 1, lower bound: -23.7268970, upper bound: 23.7260453
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 1, lower bound: -23.7257269, upper bound: 23.7259862
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 1, lower bound: -23.7256369, upper bound: 23.7256369
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 1, lower bound: -23.7257269, upper bound: 23.7259863
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 21.66
Output dim: 1, lower bound: -23.7256369, upper bound: 23.7256369

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -18.3321018, 16.4724197, -17.8733521, 16.0837612, -34.4158554, 34.3457718
1: -17.8070641, 11.2683516, -17.4196453, 11.0416307, -28.8486919, 28.6879959
2: -21.4134808, 13.8872299, -20.9635658, 13.6013508, -35.0148239, 34.8507957
3: -25.3021870, 12.1784086, -24.6895714, 11.9184494, -37.2206345, 36.8679810
4: -23.2421494, 14.8370304, -22.6677151, 14.4875212, -37.7296715, 37.5047379
5: -17.8455658, 15.9119110, -17.4307899, 15.5285530, -33.3741150, 33.3427010
6: -18.7605457, 17.1258030, -18.3233433, 16.7407608, -35.5013046, 35.4491348
7: -22.6939640, 16.3989544, -22.1887779, 16.0426064, -38.7365723, 38.5877266
8: -27.0758381, 13.7391682, -26.4377346, 13.4440279, -40.5198669, 40.1769028
9: -16.5339622, 18.5927963, -16.1577187, 18.1734924, -34.7074547, 34.7505150

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7282739, upper bound: 23.7288982
time: 6.58 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7282533, upper bound: 23.7288115
time: 18.70 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -18.1525631, 16.3317986, -18.3535156, 16.5113792, -34.6639404, 34.6853142
1: -17.6874924, 11.2259588, -17.8811073, 11.3622904, -29.0497818, 29.1070576
2: -21.3059540, 13.8146563, -21.5553150, 13.9700518, -35.2759972, 35.3699722
3: -25.0754776, 12.1024179, -25.3515930, 12.2366562, -37.3121338, 37.4540024
4: -23.0169716, 14.7195148, -23.2681942, 14.8888998, -37.9058723, 37.9877090
5: -17.7097511, 15.7650919, -17.9131336, 15.9358969, -33.6456490, 33.6782227
6: -18.6111660, 17.0028210, -18.8194771, 17.1933479, -35.8045120, 35.8222885
7: -22.5376148, 16.2972794, -22.7875481, 16.4832325, -39.0208397, 39.0848274
8: -26.8395653, 13.6633224, -27.1288662, 13.8255348, -40.6650963, 40.7921867
9: -16.4186325, 18.4559326, -16.6105728, 18.6595650, -35.0781898, 35.0665054

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7309434, upper bound: 23.7310344
time: 8.20 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7309243, upper bound: 23.7309238
time: 29.47 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -17.3692646, 15.6737442, -20.9418240, 18.8884125, -36.2576752, 36.6155663
1: -16.9533157, 10.6281815, -20.3236523, 12.6662102, -29.6195259, 30.9518261
2: -20.4045181, 13.2382202, -24.6047478, 15.8786469, -36.2831497, 37.8429680
3: -24.1203957, 11.6017513, -29.1977139, 13.9115496, -38.0319405, 40.7994652
4: -22.1258354, 14.0880680, -26.6610832, 16.9253883, -39.0512238, 40.7491531
5: -16.9396076, 15.1387300, -20.4215546, 18.2349224, -35.1745262, 35.5602837
6: -17.8737659, 16.2871666, -21.5652122, 19.5724106, -37.4461670, 37.8523788
7: -21.6860161, 15.5989590, -26.1750889, 18.7458191, -40.4318314, 41.7740440
8: -25.8537960, 13.0094776, -31.2159843, 15.5752382, -41.4290314, 44.2254639
9: -15.6846657, 17.7135735, -18.8482075, 21.3210945, -37.0057602, 36.5617752

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7268969, upper bound: 23.7260453
time: 5.60 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7268969, upper bound: 23.7260453
time: 6.34 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -18.4164734, 16.6206207, -20.7016754, 18.6858196, -37.1022949, 37.3222961
1: -17.9434052, 11.2173128, -20.0996094, 12.4992113, -30.4426155, 31.3169212
2: -21.6473083, 14.0173149, -24.3258858, 15.7024021, -37.3497086, 38.3432007
3: -25.6364517, 12.2780638, -28.8921890, 13.7580547, -39.3945084, 41.1702538
4: -23.4681358, 14.9219751, -26.3763924, 16.7339821, -40.2021179, 41.2983627
5: -17.9636307, 16.0527630, -20.1940975, 18.0405312, -36.0041618, 36.2468605
6: -18.9667034, 17.2558289, -21.3333702, 19.3517113, -38.3184128, 38.5891953
7: -23.0195160, 16.5268917, -25.9039421, 18.5367622, -41.5562782, 42.4308319
8: -27.4456196, 13.7507763, -30.8941231, 15.3815556, -42.8271751, 44.6448975
9: -16.6162128, 18.7758884, -18.6275444, 21.0902710, -37.7064819, 37.4034348

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 204

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7268969, upper bound: 23.7260453
time: 6.82 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7268969, upper bound: 23.7260453
time: 7.26 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -20.9418240, 18.8884125, -17.3692646, 15.6737442, -36.6155663, 36.2576752
1: -20.3236523, 12.6662102, -16.9533157, 10.6281815, -30.9518299, 29.6195259
2: -24.6047478, 15.8786469, -20.4045181, 13.2382202, -37.8429604, 36.2831535
3: -29.1977139, 13.9115496, -24.1203957, 11.6017513, -40.7994652, 38.0319405
4: -26.6610832, 16.9253883, -22.1258354, 14.0880680, -40.7491531, 39.0512238
5: -20.4215546, 18.2349224, -16.9396076, 15.1387300, -35.5602837, 35.1745300
6: -21.5652122, 19.5724106, -17.8737659, 16.2871666, -37.8523788, 37.4461708
7: -26.1750889, 18.7458191, -21.6860161, 15.5989590, -41.7740479, 40.4318314
8: -31.2159843, 15.5752382, -25.8537960, 13.0094776, -44.2254639, 41.4290314
9: -18.8482075, 21.3210945, -15.6846657, 17.7135735, -36.5617828, 37.0057602

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7260453, upper bound: 23.7268969
time: 12.29 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7260453, upper bound: 23.7268970
time: 5.00 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -20.7016754, 18.6858196, -18.4164734, 16.6206207, -37.3222961, 37.1022949
1: -20.0996094, 12.4992113, -17.9434052, 11.2173128, -31.3169193, 30.4426155
2: -24.3258858, 15.7024021, -21.6473083, 14.0173149, -38.3432007, 37.3497047
3: -28.8921890, 13.7580547, -25.6364517, 12.2780638, -41.1702538, 39.3945084
4: -26.3763924, 16.7339821, -23.4681358, 14.9219751, -41.2983627, 40.2021179
5: -20.1940975, 18.0405312, -17.9636307, 16.0527630, -36.2468567, 36.0041618
6: -21.3333702, 19.3517113, -18.9667034, 17.2558289, -38.5891991, 38.3184128
7: -25.9039421, 18.5367622, -23.0195160, 16.5268917, -42.4308243, 41.5562782
8: -30.8941231, 15.3815556, -27.4456196, 13.7507763, -44.6448975, 42.8271751
9: -18.6275444, 21.0902710, -16.6162128, 18.7758884, -37.4034309, 37.7064819

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7260453, upper bound: 23.7268970
time: 6.72 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7260453, upper bound: 23.7268970
time: 5.16 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -20.9418240, 18.8884125, -20.3278828, 18.3706226, -39.3124390, 39.2162933
1: -20.3236523, 12.6662102, -19.7430267, 12.2186079, -32.5422592, 32.4092369
2: -24.6047478, 15.8786469, -23.8794365, 15.4254999, -40.0302467, 39.7580681
3: -29.1977139, 13.9115496, -28.4309349, 13.5198212, -42.7175331, 42.3424797
4: -26.6610832, 16.9253883, -25.9437313, 16.4299507, -43.0910339, 42.8691177
5: -20.4215546, 18.2349224, -19.8364220, 17.7455425, -38.1670990, 38.0713425
6: -21.5652122, 19.5724106, -20.9748764, 19.0031471, -40.5683556, 40.5472870
7: -26.1750889, 18.7458191, -25.4828606, 18.2057991, -44.3808823, 44.2286797
8: -31.2159843, 15.5752382, -30.4038830, 15.0721073, -46.2880936, 45.9791145
9: -18.8482075, 21.3210945, -18.2793293, 20.7312469, -39.5794525, 39.6004257

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 107

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7256369, upper bound: 23.7256369
time: 5.50 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7256369, upper bound: 23.7256369
time: 4.10 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -20.7016754, 18.6858196, -21.3484077, 19.2951736, -39.9968452, 40.0342255
1: -20.0996094, 12.4992113, -20.7054253, 12.7872448, -32.8868523, 33.2046318
2: -24.3258858, 15.7024021, -25.0889778, 16.1840324, -40.5099106, 40.7913704
3: -28.8921890, 13.7580547, -29.9156380, 14.1793909, -43.0715752, 43.6736908
4: -26.3763924, 16.7339821, -27.2571869, 17.2418175, -43.6182060, 43.9911690
5: -20.1940975, 18.0405312, -20.8354187, 18.6389713, -38.8330688, 38.8759499
6: -21.3333702, 19.3517113, -22.0417328, 19.9448051, -41.2781754, 41.3934441
7: -25.9039421, 18.5367622, -26.7864532, 19.1096115, -45.0135460, 45.3232155
8: -30.8941231, 15.3815556, -31.9606781, 15.7906036, -46.6847229, 47.3422318
9: -18.6275444, 21.0902710, -19.1846695, 21.7682590, -40.3958015, 40.2749405

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7256369, upper bound: 23.7256369
time: 7.12 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7256369, upper bound: 23.7256369
time: 7.15 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 16.50 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.50
Output dim: 1, lower bound: -23.7282739, upper bound: 23.7288982
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.50
Output dim: 1, lower bound: -23.7282533, upper bound: 23.7288115
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.50
Output dim: 1, lower bound: -23.7309434, upper bound: 23.7310344
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.50
Output dim: 1, lower bound: -23.7309243, upper bound: 23.7309238
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.50
Output dim: 1, lower bound: -23.7268969, upper bound: 23.7260453
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.50
Output dim: 1, lower bound: -23.7268969, upper bound: 23.7260453
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.50
Output dim: 1, lower bound: -23.7268969, upper bound: 23.7260453
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.50
Output dim: 1, lower bound: -23.7268969, upper bound: 23.7260453
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 16.50
Output dim: 1, lower bound: -23.7260453, upper bound: 23.7268969
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 16.50
Output dim: 1, lower bound: -23.7260453, upper bound: 23.7268970
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 16.50
Output dim: 1, lower bound: -23.7260453, upper bound: 23.7268970
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 16.50
Output dim: 1, lower bound: -23.7260453, upper bound: 23.7268970
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 16.50
Output dim: 1, lower bound: -23.7256369, upper bound: 23.7256369
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 16.50
Output dim: 1, lower bound: -23.7256369, upper bound: 23.7256369
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 16.50
Output dim: 1, lower bound: -23.7256369, upper bound: 23.7256369
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 16.50
Output dim: 1, lower bound: -23.7256369, upper bound: 23.7256369

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -18.0435028, 16.2289619, -16.9228649, 15.2777462, -33.3212471, 33.1518250
1: -17.5348072, 11.0539751, -16.5180283, 10.3320704, -27.8668766, 27.5720024
2: -21.0792656, 13.6761789, -19.8537254, 12.8969164, -33.9761772, 33.5299034
3: -24.9419594, 11.9934177, -23.4977283, 11.3042049, -36.2461586, 35.4911385
4: -22.9114399, 14.6011629, -21.5636158, 13.7180309, -36.6294708, 36.1647720
5: -17.5589848, 15.6809969, -16.4960632, 14.7591610, -32.3181458, 32.1770554
6: -18.4858418, 16.8619194, -17.4120598, 15.8635941, -34.3494339, 34.2739792
7: -22.3732681, 16.1406841, -21.1237888, 15.1918507, -37.5651131, 37.2644691
8: -26.7041492, 13.5014763, -25.2060108, 12.6562462, -39.3603973, 38.7074852
9: -16.2648201, 18.3169289, -15.2692499, 17.2572021, -33.5220222, 33.5861778

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 64

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7282533, upper bound: 23.7288109
time: 8.03 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7282533, upper bound: 23.7288115
time: 8.39 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -17.7903709, 16.0140533, -17.9721832, 16.2268581, -34.0172272, 33.9862289
1: -17.2988434, 10.8752699, -17.5106487, 10.9221163, -28.2209587, 28.3859177
2: -20.7885857, 13.4919052, -21.0991974, 13.6777554, -34.4663391, 34.5911026
3: -24.6222420, 11.8315840, -25.0178299, 11.9824915, -36.6047325, 36.8494148
4: -22.6161556, 14.3956623, -22.9092026, 14.5538225, -37.1699753, 37.3048553
5: -17.3092823, 15.4762630, -17.5225525, 15.6755199, -32.9848022, 32.9988174
6: -18.2432003, 16.6308727, -18.5075836, 16.8345699, -35.0777702, 35.1384583
7: -22.0892601, 15.9171276, -22.4607048, 16.1220112, -38.2112694, 38.3778191
8: -26.3698769, 13.2956753, -26.8016701, 13.3995152, -39.7693939, 40.0973434
9: -16.0308819, 18.0746899, -16.2026997, 18.3222218, -34.3530998, 34.2773895

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 64

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7282536, upper bound: 23.7288115
time: 6.78 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7282533, upper bound: 23.7288109
time: 4.45 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -17.8558102, 16.0806694, -17.3692646, 15.6737442, -33.5295563, 33.4499359
1: -17.4114532, 11.0053768, -16.9533157, 10.6281815, -28.0396347, 27.9586906
2: -20.9596786, 13.5946379, -20.4045181, 13.2382202, -34.1978989, 33.9991570
3: -24.7056217, 11.9122877, -24.1203957, 11.6017513, -36.3073730, 36.0326843
4: -22.6760216, 14.4770451, -22.1258354, 14.0880680, -36.7640915, 36.6028824
5: -17.4146957, 15.5274839, -16.9396076, 15.1387300, -32.5534248, 32.4670906
6: -18.3267689, 16.7314320, -17.8737659, 16.2871666, -34.6139336, 34.6051979
7: -22.2074814, 16.0307121, -21.6860161, 15.5989590, -37.8064423, 37.7167168
8: -26.4568958, 13.4176064, -25.8537960, 13.0094776, -39.4663734, 39.2714005
9: -16.1394825, 18.1732998, -15.6846657, 17.7135735, -33.8530502, 33.8579636

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 85

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7309238, upper bound: 23.7309243
time: 9.43 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7309238, upper bound: 23.7309238
time: 15.00 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -17.5992985, 15.8625994, -18.4164734, 16.6206207, -34.2199173, 34.2790680
1: -17.1748638, 10.8257580, -17.9434052, 11.2173128, -28.3921776, 28.7691631
2: -20.6654701, 13.4069118, -21.6473083, 14.0173149, -34.6827850, 35.0542183
3: -24.3825169, 11.7486286, -25.6364517, 12.2780638, -36.6605797, 37.3850746
4: -22.3757782, 14.2700176, -23.4681358, 14.9219751, -37.2977524, 37.7381516
5: -17.1631737, 15.3195887, -17.9636307, 16.0527630, -33.2159348, 33.2832184
6: -18.0813065, 16.4976807, -18.9667034, 17.2558289, -35.3371315, 35.4643822
7: -21.9207764, 15.8041401, -23.0195160, 16.5268917, -38.4476700, 38.8236542
8: -26.1188965, 13.2096481, -27.4456196, 13.7507763, -39.8696747, 40.6552658
9: -15.9023838, 17.9288788, -16.6162128, 18.7758884, -34.6782722, 34.5450897

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 64

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7309243, upper bound: 23.7309238
time: 8.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7309243, upper bound: 23.7309238
time: 5.58 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -17.3692646, 15.6737442, -20.3278828, 18.3706226, -35.7398872, 36.0016251
1: -16.9533157, 10.6281815, -19.7430267, 12.2186079, -29.1719246, 30.3712082
2: -20.4045181, 13.2382202, -23.8794365, 15.4254999, -35.8300171, 37.1176491
3: -24.1203957, 11.6017513, -28.4309349, 13.5198212, -37.6402168, 40.0326843
4: -22.1258354, 14.0880680, -25.9437313, 16.4299507, -38.5557861, 40.0317993
5: -16.9396076, 15.1387300, -19.8364220, 17.7455425, -34.6851501, 34.9751511
6: -17.8737659, 16.2871666, -20.9748764, 19.0031471, -36.8769035, 37.2620430
7: -21.6860161, 15.5989590, -25.4828606, 18.2057991, -39.8918037, 41.0818176
8: -25.8537960, 13.0094776, -30.4038830, 15.0721073, -40.9259033, 43.4133530
9: -15.6846657, 17.7135735, -18.2793293, 20.7312469, -36.4159126, 35.9928970

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7188918, upper bound: 23.7185319
time: 14.69 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7270141, upper bound: 23.7260778
time: 10.55 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -17.3692646, 15.6737442, -21.3484077, 19.2951736, -36.6644363, 37.0221519
1: -16.9533157, 10.6281815, -20.7054253, 12.7872448, -29.7405605, 31.3335953
2: -20.4045181, 13.2382202, -25.0889778, 16.1840324, -36.5885468, 38.3271904
3: -24.1203957, 11.6017513, -29.9156380, 14.1793909, -38.2997856, 41.5173874
4: -22.1258354, 14.0880680, -27.2571869, 17.2418175, -39.3676529, 41.3452530
5: -16.9396076, 15.1387300, -20.8354187, 18.6389713, -35.5785789, 35.9741478
6: -17.8737659, 16.2871666, -22.0417328, 19.9448051, -37.8185654, 38.3288956
7: -21.6860161, 15.5989590, -26.7864532, 19.1096115, -40.7956161, 42.3854103
8: -25.8537960, 13.0094776, -31.9606781, 15.7906036, -41.6443939, 44.9701538
9: -15.6846657, 17.7135735, -19.1846695, 21.7682590, -37.4529266, 36.8982353

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7262516, upper bound: 23.7254485
time: 10.31 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7268188, upper bound: 23.7259020
time: 5.74 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -18.4164734, 16.6206207, -20.3278828, 18.3706226, -36.7870941, 36.9485016
1: -17.9434052, 11.2173128, -19.7430267, 12.2186079, -30.1620140, 30.9603386
2: -21.6473083, 14.0173149, -23.8794365, 15.4254999, -37.0728073, 37.8967438
3: -25.6364517, 12.2780638, -28.4309349, 13.5198212, -39.1562729, 40.7089996
4: -23.4681358, 14.9219751, -25.9437313, 16.4299507, -39.8980865, 40.8656998
5: -17.9636307, 16.0527630, -19.8364220, 17.7455425, -35.7091751, 35.8891830
6: -18.9667034, 17.2558289, -20.9748764, 19.0031471, -37.9698448, 38.2307053
7: -23.0195160, 16.5268917, -25.4828606, 18.2057991, -41.2253113, 42.0097504
8: -27.4456196, 13.7507763, -30.4038830, 15.0721073, -42.5177269, 44.1546593
9: -16.6162128, 18.7758884, -18.2793293, 20.7312469, -37.3474579, 37.0552177

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7177973, upper bound: 23.7175176
time: 8.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7268969, upper bound: 23.7260453
time: 8.55 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -18.4164734, 16.6206207, -21.3484077, 19.2951736, -37.7116432, 37.9690285
1: -17.9434052, 11.2173128, -20.7054253, 12.7872448, -30.7306499, 31.9227295
2: -21.6473083, 14.0173149, -25.0889778, 16.1840324, -37.8313370, 39.1062927
3: -25.6364517, 12.2780638, -29.9156380, 14.1793909, -39.8158417, 42.1937027
4: -23.4681358, 14.9219751, -27.2571869, 17.2418175, -40.7099533, 42.1791534
5: -17.9636307, 16.0527630, -20.8354187, 18.6389713, -36.6026001, 36.8881798
6: -18.9667034, 17.2558289, -22.0417328, 19.9448051, -38.9115067, 39.2975578
7: -23.0195160, 16.5268917, -26.7864532, 19.1096115, -42.1291237, 43.3133392
8: -27.4456196, 13.7507763, -31.9606781, 15.7906036, -43.2362213, 45.7114563
9: -16.6162128, 18.7758884, -19.1846695, 21.7682590, -38.3844719, 37.9605560

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7177973, upper bound: 23.7175171
time: 10.74 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7268969, upper bound: 23.7260453
time: 4.86 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -20.3278828, 18.3706226, -17.3692646, 15.6737442, -36.0016251, 35.7398872
1: -19.7430267, 12.2186079, -16.9533157, 10.6281815, -30.3712082, 29.1719246
2: -23.8794365, 15.4254999, -20.4045181, 13.2382202, -37.1176529, 35.8300171
3: -28.4309349, 13.5198212, -24.1203957, 11.6017513, -40.0326843, 37.6402168
4: -25.9437313, 16.4299507, -22.1258354, 14.0880680, -40.0317993, 38.5557861
5: -19.8364220, 17.7455425, -16.9396076, 15.1387300, -34.9751511, 34.6851501
6: -20.9748764, 19.0031471, -17.8737659, 16.2871666, -37.2620392, 36.8769035
7: -25.4828606, 18.2057991, -21.6860161, 15.5989590, -41.0818176, 39.8918037
8: -30.4038830, 15.0721073, -25.8537960, 13.0094776, -43.4133530, 40.9259033
9: -18.2793293, 20.7312469, -15.6846657, 17.7135735, -35.9929047, 36.4159126

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7185320, upper bound: 23.7188918
time: 13.74 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7260778, upper bound: 23.7270141
time: 10.17 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -21.3484077, 19.2951736, -17.3692646, 15.6737442, -37.0221519, 36.6644363
1: -20.7054253, 12.7872448, -16.9533157, 10.6281815, -31.3335972, 29.7405605
2: -25.0889778, 16.1840324, -20.4045181, 13.2382202, -38.3271980, 36.5885468
3: -29.9156380, 14.1793909, -24.1203957, 11.6017513, -41.5173874, 38.2997818
4: -27.2571869, 17.2418175, -22.1258354, 14.0880680, -41.3452530, 39.3676529
5: -20.8354187, 18.6389713, -16.9396076, 15.1387300, -35.9741478, 35.5785751
6: -22.0417328, 19.9448051, -17.8737659, 16.2871666, -38.3288918, 37.8185692
7: -26.7864532, 19.1096115, -21.6860161, 15.5989590, -42.3854141, 40.7956161
8: -31.9606781, 15.7906036, -25.8537960, 13.0094776, -44.9701538, 41.6443939
9: -19.1846695, 21.7682590, -15.6846657, 17.7135735, -36.8982391, 37.4529266

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7254485, upper bound: 23.7262516
time: 6.88 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7259020, upper bound: 23.7268188
time: 23.99 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -20.3278828, 18.3706226, -18.4164734, 16.6206207, -36.9485016, 36.7870941
1: -19.7430267, 12.2186079, -17.9434052, 11.2173128, -30.9603386, 30.1620140
2: -23.8794365, 15.4254999, -21.6473083, 14.0173149, -37.8967514, 37.0728073
3: -28.4309349, 13.5198212, -25.6364517, 12.2780638, -40.7089996, 39.1562729
4: -25.9437313, 16.4299507, -23.4681358, 14.9219751, -40.8657074, 39.8980865
5: -19.8364220, 17.7455425, -17.9636307, 16.0527630, -35.8891830, 35.7091751
6: -20.9748764, 19.0031471, -18.9667034, 17.2558289, -38.2307053, 37.9698448
7: -25.4828606, 18.2057991, -23.0195160, 16.5268917, -42.0097504, 41.2253151
8: -30.4038830, 15.0721073, -27.4456196, 13.7507763, -44.1546593, 42.5177269
9: -18.2793293, 20.7312469, -16.6162128, 18.7758884, -37.0552177, 37.3474579

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7175172, upper bound: 23.7177973
time: 10.28 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7260453, upper bound: 23.7268970
time: 6.21 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -21.3484077, 19.2951736, -18.4164734, 16.6206207, -37.9690285, 37.7116394
1: -20.7054253, 12.7872448, -17.9434052, 11.2173128, -31.9227352, 30.7306499
2: -25.0889778, 16.1840324, -21.6473083, 14.0173149, -39.1062927, 37.8313370
3: -29.9156380, 14.1793909, -25.6364517, 12.2780638, -42.1937027, 39.8158417
4: -27.2571869, 17.2418175, -23.4681358, 14.9219751, -42.1791534, 40.7099533
5: -20.8354187, 18.6389713, -17.9636307, 16.0527630, -36.8881798, 36.6026001
6: -22.0417328, 19.9448051, -18.9667034, 17.2558289, -39.2975578, 38.9115067
7: -26.7864532, 19.1096115, -23.0195160, 16.5268917, -43.3133469, 42.1291237
8: -31.9606781, 15.7906036, -27.4456196, 13.7507763, -45.7114563, 43.2362213
9: -19.1846695, 21.7682590, -16.6162128, 18.7758884, -37.9605560, 38.3844719

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7175172, upper bound: 23.7177977
time: 6.84 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -23.7260453, upper bound: 23.7268970
time: 10.36 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 14.63 + 591.93 = 606.55 seconds
