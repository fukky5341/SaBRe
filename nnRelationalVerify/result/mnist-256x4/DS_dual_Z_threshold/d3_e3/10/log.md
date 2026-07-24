## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 105.986448459


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352)
1: (-48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901)
2: (-63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453)
3: (-66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463)
4: (-61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134)
5: (-55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531)
6: (-53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280)
7: (-57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380)
8: (-70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773)
9: (-53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.14 + 11.40 = 13.55 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -106.0925410, upper bound: 106.0925410

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0857212, upper bound: 106.0857211
time: 8.34 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0857212, upper bound: 106.0857212
time: 8.52 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 17.06 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 17.06
Output dim: 7, lower bound: -106.0857212, upper bound: 106.0857211
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 17.06
Output dim: 7, lower bound: -106.0857212, upper bound: 106.0857212

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0734149, upper bound: 106.0734149
time: 7.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0734149, upper bound: 106.0734149
time: 7.78 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0734149, upper bound: 106.0734149
time: 7.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0734149, upper bound: 106.0734149
time: 8.04 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 18.28 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 18.28
Output dim: 7, lower bound: -106.0734149, upper bound: 106.0734149
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 18.28
Output dim: 7, lower bound: -106.0734149, upper bound: 106.0734149
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 18.28
Output dim: 7, lower bound: -106.0734149, upper bound: 106.0734149
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 18.28
Output dim: 7, lower bound: -106.0734149, upper bound: 106.0734149

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0672889, upper bound: 106.0672889
time: 7.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0672889, upper bound: 106.0672889
time: 6.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0672889, upper bound: 106.0672889
time: 7.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0672889, upper bound: 106.0672889
time: 6.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0672889, upper bound: 106.0672889
time: 7.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0672889, upper bound: 106.0672889
time: 6.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0672889, upper bound: 106.0672889
time: 7.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0672889, upper bound: 106.0672889
time: 6.66 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 16.22 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.22
Output dim: 7, lower bound: -106.0672889, upper bound: 106.0672889
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.22
Output dim: 7, lower bound: -106.0672889, upper bound: 106.0672889
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.22
Output dim: 7, lower bound: -106.0672889, upper bound: 106.0672889
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.22
Output dim: 7, lower bound: -106.0672889, upper bound: 106.0672889
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.22
Output dim: 7, lower bound: -106.0672889, upper bound: 106.0672889
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.22
Output dim: 7, lower bound: -106.0672889, upper bound: 106.0672889
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.22
Output dim: 7, lower bound: -106.0672889, upper bound: 106.0672889
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.22
Output dim: 7, lower bound: -106.0672889, upper bound: 106.0672889

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0617273, upper bound: 106.0617259
time: 6.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0617317, upper bound: 106.0617233
time: 7.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0617233, upper bound: 106.0617317
time: 8.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0617259, upper bound: 106.0617273
time: 6.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0617273, upper bound: 106.0617259
time: 6.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0617317, upper bound: 106.0617233
time: 7.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0617233, upper bound: 106.0617317
time: 7.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0617259, upper bound: 106.0617273
time: 7.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0617273, upper bound: 106.0617259
time: 7.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0617317, upper bound: 106.0617233
time: 6.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0617233, upper bound: 106.0617317
time: 7.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0617259, upper bound: 106.0617273
time: 6.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0617273, upper bound: 106.0617259
time: 7.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0617317, upper bound: 106.0617233
time: 6.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0617233, upper bound: 106.0617317
time: 7.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0617259, upper bound: 106.0617273
time: 7.40 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 22.00 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 7, lower bound: -106.0617273, upper bound: 106.0617259
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 7, lower bound: -106.0617317, upper bound: 106.0617233
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 7, lower bound: -106.0617233, upper bound: 106.0617317
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 7, lower bound: -106.0617259, upper bound: 106.0617273
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 7, lower bound: -106.0617273, upper bound: 106.0617259
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 7, lower bound: -106.0617317, upper bound: 106.0617233
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 7, lower bound: -106.0617233, upper bound: 106.0617317
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 7, lower bound: -106.0617259, upper bound: 106.0617273
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 7, lower bound: -106.0617273, upper bound: 106.0617259
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 7, lower bound: -106.0617317, upper bound: 106.0617233
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 7, lower bound: -106.0617233, upper bound: 106.0617317
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 7, lower bound: -106.0617259, upper bound: 106.0617273
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 7, lower bound: -106.0617273, upper bound: 106.0617259
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 7, lower bound: -106.0617317, upper bound: 106.0617233
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 7, lower bound: -106.0617233, upper bound: 106.0617317
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 7, lower bound: -106.0617259, upper bound: 106.0617273

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606839, upper bound: 106.0606787
time: 8.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606841, upper bound: 106.0606785
time: 7.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606881, upper bound: 106.0606776
time: 7.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606881, upper bound: 106.0606776
time: 7.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606776, upper bound: 106.0606881
time: 7.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606776, upper bound: 106.0606881
time: 6.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606785, upper bound: 106.0606841
time: 7.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606787, upper bound: 106.0606839
time: 7.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606839, upper bound: 106.0606787
time: 8.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606841, upper bound: 106.0606785
time: 6.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606881, upper bound: 106.0606776
time: 7.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606881, upper bound: 106.0606776
time: 7.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606776, upper bound: 106.0606881
time: 7.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606776, upper bound: 106.0606881
time: 6.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606785, upper bound: 106.0606841
time: 7.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606787, upper bound: 106.0606839
time: 7.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606839, upper bound: 106.0606787
time: 8.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606841, upper bound: 106.0606785
time: 7.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606881, upper bound: 106.0606776
time: 7.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606881, upper bound: 106.0606776
time: 7.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606776, upper bound: 106.0606881
time: 7.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606776, upper bound: 106.0606881
time: 7.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606785, upper bound: 106.0606841
time: 7.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606787, upper bound: 106.0606839
time: 7.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606839, upper bound: 106.0606787
time: 7.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606841, upper bound: 106.0606785
time: 7.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -57.8715248, 46.4339218, -57.8715248, 46.4339218, -104.3054352, 104.3054352
1: -48.2762108, 41.0486870, -48.2762108, 41.0486870, -89.3248901, 89.3248901
2: -63.9665565, 41.6315002, -63.9665565, 41.6315002, -105.5980453, 105.5980453
3: -66.9424515, 36.1956024, -66.9424515, 36.1956024, -103.1380463, 103.1380463
4: -61.7093658, 47.3356514, -61.7093658, 47.3356514, -109.0450134, 109.0450134
5: -55.6309967, 43.3028641, -55.6309967, 43.3028641, -98.9338531, 98.9338531
6: -53.1199455, 51.6415939, -53.1199455, 51.6415939, -104.7615356, 104.7615280
7: -57.6861839, 49.2933502, -57.6861839, 49.2933502, -106.9795380, 106.9795380
8: -70.8696365, 48.8952446, -70.8696365, 48.8952446, -119.7648773, 119.7648773
9: -53.1959190, 52.2425995, -53.1959190, 52.2425995, -105.4385071, 105.4385071

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606881, upper bound: 106.0606776
time: 7.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606881, upper bound: 106.0606776
time: 7.25 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 20.91 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606839, upper bound: 106.0606787
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606841, upper bound: 106.0606785
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606881, upper bound: 106.0606776
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606881, upper bound: 106.0606776
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606776, upper bound: 106.0606881
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606776, upper bound: 106.0606881
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606785, upper bound: 106.0606841
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606787, upper bound: 106.0606839
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606839, upper bound: 106.0606787
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606841, upper bound: 106.0606785
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606881, upper bound: 106.0606776
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606881, upper bound: 106.0606776
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606776, upper bound: 106.0606881
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606776, upper bound: 106.0606881
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606785, upper bound: 106.0606841
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606787, upper bound: 106.0606839
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606839, upper bound: 106.0606787
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606841, upper bound: 106.0606785
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606881, upper bound: 106.0606776
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606881, upper bound: 106.0606776
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606776, upper bound: 106.0606881
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606776, upper bound: 106.0606881
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606785, upper bound: 106.0606841
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606787, upper bound: 106.0606839
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606839, upper bound: 106.0606787
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606841, upper bound: 106.0606785
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606881, upper bound: 106.0606776
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.91
Output dim: 7, lower bound: -106.0606881, upper bound: 106.0606776
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 20.91
Output dim: 7, lower bound: -106.0617233, upper bound: 106.0617317
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 20.91
Output dim: 7, lower bound: -106.0617259, upper bound: 106.0617273

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 13.55 + 587.97 = 601.51 seconds
