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
execution time: IAR + RelationalAnalysis = 0.85 + 11.10 = 11.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -106.0925410, upper bound: 106.0925410

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0925376, upper bound: 106.0925410
time: 7.62 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0925410, upper bound: 106.0925376
time: 7.54 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 15.18 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 15.18
Output dim: 7, lower bound: -106.0925376, upper bound: 106.0925410
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 15.18
Output dim: 7, lower bound: -106.0925410, upper bound: 106.0925376

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0770421, upper bound: 106.0770451
time: 6.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0770421, upper bound: 106.0770451
time: 6.90 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0885550, upper bound: 106.0885540
time: 8.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0885550, upper bound: 106.0885539
time: 8.39 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 17.53 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 17.53
Output dim: 7, lower bound: -106.0770421, upper bound: 106.0770451
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 17.53
Output dim: 7, lower bound: -106.0770421, upper bound: 106.0770451
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 17.53
Output dim: 7, lower bound: -106.0885550, upper bound: 106.0885540
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 17.53
Output dim: 7, lower bound: -106.0885550, upper bound: 106.0885539

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0724776, upper bound: 106.0724773
time: 6.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0724776, upper bound: 106.0724773
time: 6.42 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0650526, upper bound: 106.0650543
time: 7.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0650526, upper bound: 106.0650543
time: 7.34 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0885550, upper bound: 106.0885539
time: 8.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0885550, upper bound: 106.0885540
time: 7.49 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0885168, upper bound: 106.0885135
time: 7.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0885150, upper bound: 106.0885161
time: 7.37 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 15.94 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 7, lower bound: -106.0724776, upper bound: 106.0724773
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 7, lower bound: -106.0724776, upper bound: 106.0724773
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 7, lower bound: -106.0650526, upper bound: 106.0650543
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 7, lower bound: -106.0650526, upper bound: 106.0650543
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 7, lower bound: -106.0885550, upper bound: 106.0885539
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 7, lower bound: -106.0885550, upper bound: 106.0885540
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 7, lower bound: -106.0885168, upper bound: 106.0885135
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 7, lower bound: -106.0885150, upper bound: 106.0885161

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0719762, upper bound: 106.0719630
time: 7.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0719632, upper bound: 106.0719762
time: 6.31 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 162

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0709501, upper bound: 106.0709599
time: 6.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0709501, upper bound: 106.0709599
time: 7.19 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0649320, upper bound: 106.0649357
time: 6.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0649317, upper bound: 106.0649357
time: 6.26 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0650474, upper bound: 106.0650543
time: 6.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0650526, upper bound: 106.0650494
time: 6.18 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0819010, upper bound: 106.0818996
time: 6.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0819010, upper bound: 106.0818996
time: 6.43 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 133

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0851037, upper bound: 106.0850982
time: 9.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0851037, upper bound: 106.0850982
time: 8.87 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 162

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0883607, upper bound: 106.0883488
time: 8.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0883501, upper bound: 106.0883582
time: 7.01 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0808803, upper bound: 106.0809021
time: 7.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0808803, upper bound: 106.0809021
time: 8.46 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 16.46 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -106.0719762, upper bound: 106.0719630
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -106.0719632, upper bound: 106.0719762
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -106.0709501, upper bound: 106.0709599
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -106.0709501, upper bound: 106.0709599
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -106.0649320, upper bound: 106.0649357
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -106.0649317, upper bound: 106.0649357
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -106.0650474, upper bound: 106.0650543
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -106.0650526, upper bound: 106.0650494
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -106.0819010, upper bound: 106.0818996
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -106.0819010, upper bound: 106.0818996
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -106.0851037, upper bound: 106.0850982
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -106.0851037, upper bound: 106.0850982
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -106.0883607, upper bound: 106.0883488
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -106.0883501, upper bound: 106.0883582
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -106.0808803, upper bound: 106.0809021
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.46
Output dim: 7, lower bound: -106.0808803, upper bound: 106.0809021

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0705815, upper bound: 106.0705680
time: 7.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0705745, upper bound: 106.0705745
time: 6.03 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0709031, upper bound: 106.0709175
time: 6.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0709041, upper bound: 106.0709163
time: 6.79 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0699854, upper bound: 106.0699913
time: 6.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0699823, upper bound: 106.0699930
time: 8.31 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0704958, upper bound: 106.0705006
time: 7.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0704917, upper bound: 106.0705040
time: 6.66 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0627926, upper bound: 106.0627680
time: 6.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0627680, upper bound: 106.0627929
time: 7.36 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 99

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606703, upper bound: 106.0606675
time: 5.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0606703, upper bound: 106.0606675
time: 5.48 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 133

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0647545, upper bound: 106.0647582
time: 6.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0647536, upper bound: 106.0647583
time: 6.85 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0649994, upper bound: 106.0649745
time: 6.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0649775, upper bound: 106.0649950
time: 6.80 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0627525, upper bound: 106.0627561
time: 5.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0627525, upper bound: 106.0627561
time: 7.31 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 210

### Candidate
type: DSZ, layer: 1, pos: 133

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0773781, upper bound: 106.0773741
time: 7.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0773781, upper bound: 106.0773741
time: 7.49 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0746256, upper bound: 106.0746252
time: 7.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0746256, upper bound: 106.0746252
time: 7.32 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0766843, upper bound: 106.0766854
time: 7.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0766843, upper bound: 106.0766854
time: 7.77 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0872047, upper bound: 106.0871878
time: 8.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0872062, upper bound: 106.0871860
time: 7.75 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0862559, upper bound: 106.0862645
time: 7.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0862559, upper bound: 106.0862645
time: 8.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0727579, upper bound: 106.0727746
time: 7.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0727579, upper bound: 106.0727746
time: 7.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0808797, upper bound: 106.0809021
time: 8.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0808803, upper bound: 106.0809014
time: 8.01 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 19.30 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0705815, upper bound: 106.0705680
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0705745, upper bound: 106.0705745
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0709031, upper bound: 106.0709175
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0709041, upper bound: 106.0709163
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0699854, upper bound: 106.0699913
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0699823, upper bound: 106.0699930
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0704958, upper bound: 106.0705006
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0704917, upper bound: 106.0705040
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0627926, upper bound: 106.0627680
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0627680, upper bound: 106.0627929
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0606703, upper bound: 106.0606675
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0606703, upper bound: 106.0606675
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0647545, upper bound: 106.0647582
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0647536, upper bound: 106.0647583
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0649994, upper bound: 106.0649745
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0649775, upper bound: 106.0649950
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0627525, upper bound: 106.0627561
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0627525, upper bound: 106.0627561
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0773781, upper bound: 106.0773741
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0773781, upper bound: 106.0773741
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0746256, upper bound: 106.0746252
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0746256, upper bound: 106.0746252
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0766843, upper bound: 106.0766854
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0766843, upper bound: 106.0766854
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0872047, upper bound: 106.0871878
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0872062, upper bound: 106.0871860
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0862559, upper bound: 106.0862645
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0862559, upper bound: 106.0862645
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0727579, upper bound: 106.0727746
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0727579, upper bound: 106.0727746
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0808797, upper bound: 106.0809021
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.30
Output dim: 7, lower bound: -106.0808803, upper bound: 106.0809014

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 208

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0690182, upper bound: 106.0689875
time: 6.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0689998, upper bound: 106.0690034
time: 7.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 214

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0695509, upper bound: 106.0695305
time: 6.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0695299, upper bound: 106.0695448
time: 6.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0686539, upper bound: 106.0686243
time: 6.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0686231, upper bound: 106.0686689
time: 6.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0708985, upper bound: 106.0709163
time: 6.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0709041, upper bound: 106.0709137
time: 6.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0699854, upper bound: 106.0699659
time: 7.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0699600, upper bound: 106.0699913
time: 6.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 208

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0684987, upper bound: 106.0685004
time: 6.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -106.0684951, upper bound: 106.0685073
time: 6.78 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 14.11 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.11
Output dim: 7, lower bound: -106.0690182, upper bound: 106.0689875
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.11
Output dim: 7, lower bound: -106.0689998, upper bound: 106.0690034
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.11
Output dim: 7, lower bound: -106.0695509, upper bound: 106.0695305
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.11
Output dim: 7, lower bound: -106.0695299, upper bound: 106.0695448
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.11
Output dim: 7, lower bound: -106.0686539, upper bound: 106.0686243
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.11
Output dim: 7, lower bound: -106.0686231, upper bound: 106.0686689
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.11
Output dim: 7, lower bound: -106.0708985, upper bound: 106.0709163
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.11
Output dim: 7, lower bound: -106.0709041, upper bound: 106.0709137
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.11
Output dim: 7, lower bound: -106.0699854, upper bound: 106.0699659
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.11
Output dim: 7, lower bound: -106.0699600, upper bound: 106.0699913
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.11
Output dim: 7, lower bound: -106.0684987, upper bound: 106.0685004
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.11
Output dim: 7, lower bound: -106.0684951, upper bound: 106.0685073
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0704958, upper bound: 106.0705006
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0704917, upper bound: 106.0705040
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0627926, upper bound: 106.0627680
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0627680, upper bound: 106.0627929
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0606703, upper bound: 106.0606675
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0606703, upper bound: 106.0606675
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0647545, upper bound: 106.0647582
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0647536, upper bound: 106.0647583
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0649994, upper bound: 106.0649745
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0649775, upper bound: 106.0649950
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0627525, upper bound: 106.0627561
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0627525, upper bound: 106.0627561
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0773781, upper bound: 106.0773741
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0773781, upper bound: 106.0773741
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0746256, upper bound: 106.0746252
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0746256, upper bound: 106.0746252
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0766843, upper bound: 106.0766854
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0766843, upper bound: 106.0766854
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0872047, upper bound: 106.0871878
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0872062, upper bound: 106.0871860
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0862559, upper bound: 106.0862645
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0862559, upper bound: 106.0862645
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0727579, upper bound: 106.0727746
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0727579, upper bound: 106.0727746
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0808797, upper bound: 106.0809021
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.11
Output dim: 7, lower bound: -106.0808803, upper bound: 106.0809014

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 11.96 + 592.13 = 604.09 seconds
