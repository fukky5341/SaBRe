## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 106.6602947382


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797)
1: (-48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317)
2: (-65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747)
3: (-68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471)
4: (-64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390)
5: (-57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115)
6: (-54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235)
7: (-58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829)
8: (-71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617)
9: (-54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.90 + 10.92 = 11.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -106.7670618, upper bound: 106.7670618

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7661850, upper bound: 106.7661850
time: 7.30 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7661850, upper bound: 106.7661850
time: 6.40 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 13.72 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 13.72
Output dim: 0, lower bound: -106.7661850, upper bound: 106.7661850
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 13.72
Output dim: 0, lower bound: -106.7661850, upper bound: 106.7661850

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7270566, upper bound: 106.7270566
time: 7.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7270566, upper bound: 106.7270566
time: 7.97 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7623734, upper bound: 106.7623699
time: 7.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7623700, upper bound: 106.7623733
time: 7.01 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 15.36 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 15.36
Output dim: 0, lower bound: -106.7270566, upper bound: 106.7270566
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 15.36
Output dim: 0, lower bound: -106.7270566, upper bound: 106.7270566
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 15.36
Output dim: 0, lower bound: -106.7623734, upper bound: 106.7623699
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 15.36
Output dim: 0, lower bound: -106.7623700, upper bound: 106.7623733

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7252472, upper bound: 106.7252475
time: 6.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7252475, upper bound: 106.7252472
time: 6.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7144169, upper bound: 106.7144169
time: 6.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7144169, upper bound: 106.7144169
time: 6.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7623733, upper bound: 106.7623699
time: 6.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7623733, upper bound: 106.7623699
time: 6.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7613951, upper bound: 106.7613988
time: 5.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7613951, upper bound: 106.7613988
time: 7.78 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 14.55 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.55
Output dim: 0, lower bound: -106.7252472, upper bound: 106.7252475
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.55
Output dim: 0, lower bound: -106.7252475, upper bound: 106.7252472
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.55
Output dim: 0, lower bound: -106.7144169, upper bound: 106.7144169
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.55
Output dim: 0, lower bound: -106.7144169, upper bound: 106.7144169
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.55
Output dim: 0, lower bound: -106.7623733, upper bound: 106.7623699
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.55
Output dim: 0, lower bound: -106.7623733, upper bound: 106.7623699
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.55
Output dim: 0, lower bound: -106.7613951, upper bound: 106.7613988
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.55
Output dim: 0, lower bound: -106.7613951, upper bound: 106.7613988

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7196357, upper bound: 106.7196358
time: 6.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7196357, upper bound: 106.7196357
time: 5.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6744395, upper bound: 106.6744395
time: 6.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6744395, upper bound: 106.6744395
time: 6.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 200

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6915427, upper bound: 106.6915422
time: 6.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6915427, upper bound: 106.6915422
time: 5.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6984166, upper bound: 106.6984166
time: 7.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6984166, upper bound: 106.6984166
time: 6.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7423469, upper bound: 106.7423502
time: 8.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7423469, upper bound: 106.7423502
time: 7.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7233875, upper bound: 106.7233857
time: 6.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7233875, upper bound: 106.7233857
time: 5.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7464887, upper bound: 106.7464878
time: 6.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7464880, upper bound: 106.7464884
time: 6.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6668761, upper bound: 106.6668769
time: 6.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6668761, upper bound: 106.6668769
time: 6.25 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 13.26 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.26
Output dim: 0, lower bound: -106.7196357, upper bound: 106.7196358
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.26
Output dim: 0, lower bound: -106.7196357, upper bound: 106.7196357
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.26
Output dim: 0, lower bound: -106.6744395, upper bound: 106.6744395
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.26
Output dim: 0, lower bound: -106.6744395, upper bound: 106.6744395
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.26
Output dim: 0, lower bound: -106.6915427, upper bound: 106.6915422
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.26
Output dim: 0, lower bound: -106.6915427, upper bound: 106.6915422
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.26
Output dim: 0, lower bound: -106.6984166, upper bound: 106.6984166
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.26
Output dim: 0, lower bound: -106.6984166, upper bound: 106.6984166
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.26
Output dim: 0, lower bound: -106.7423469, upper bound: 106.7423502
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.26
Output dim: 0, lower bound: -106.7423469, upper bound: 106.7423502
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.26
Output dim: 0, lower bound: -106.7233875, upper bound: 106.7233857
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.26
Output dim: 0, lower bound: -106.7233875, upper bound: 106.7233857
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.26
Output dim: 0, lower bound: -106.7464887, upper bound: 106.7464878
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.26
Output dim: 0, lower bound: -106.7464880, upper bound: 106.7464884
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.26
Output dim: 0, lower bound: -106.6668761, upper bound: 106.6668769
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.26
Output dim: 0, lower bound: -106.6668761, upper bound: 106.6668769

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7128951, upper bound: 106.7128989
time: 7.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7128984, upper bound: 106.7128945
time: 6.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6779835, upper bound: 106.6779834
time: 6.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6779835, upper bound: 106.6779834
time: 6.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6718065, upper bound: 106.6718063
time: 6.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6718065, upper bound: 106.6718063
time: 7.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6739336, upper bound: 106.6739328
time: 5.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6739336, upper bound: 106.6739328
time: 5.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6898181, upper bound: 106.6898127
time: 8.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6898181, upper bound: 106.6898127
time: 7.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6354769, upper bound: 106.6354766
time: 5.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6354769, upper bound: 106.6354766
time: 5.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6931161, upper bound: 106.6931160
time: 7.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6931160, upper bound: 106.6931161
time: 8.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 126

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6790179, upper bound: 106.6790161
time: 6.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6790179, upper bound: 106.6790161
time: 6.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7277343, upper bound: 106.7277391
time: 7.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7277345, upper bound: 106.7277387
time: 7.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7423469, upper bound: 106.7423502
time: 7.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7423469, upper bound: 106.7423501
time: 6.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7233875, upper bound: 106.7233828
time: 5.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7233858, upper bound: 106.7233857
time: 4.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7233875, upper bound: 106.7233850
time: 5.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7233872, upper bound: 106.7233857
time: 6.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7256768, upper bound: 106.7256764
time: 6.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7256768, upper bound: 106.7256764
time: 6.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7419713, upper bound: 106.7419693
time: 7.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7419716, upper bound: 106.7419693
time: 7.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6668761, upper bound: 106.6668765
time: 6.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6668760, upper bound: 106.6668769
time: 6.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6398038, upper bound: 106.6398057
time: 6.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6398038, upper bound: 106.6398057
time: 6.22 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 15.04 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.7128951, upper bound: 106.7128989
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.7128984, upper bound: 106.7128945
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6779835, upper bound: 106.6779834
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6779835, upper bound: 106.6779834
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6718065, upper bound: 106.6718063
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6718065, upper bound: 106.6718063
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6739336, upper bound: 106.6739328
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6739336, upper bound: 106.6739328
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6898181, upper bound: 106.6898127
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6898181, upper bound: 106.6898127
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6354769, upper bound: 106.6354766
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6354769, upper bound: 106.6354766
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6931161, upper bound: 106.6931160
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6931160, upper bound: 106.6931161
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6790179, upper bound: 106.6790161
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6790179, upper bound: 106.6790161
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.7277343, upper bound: 106.7277391
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.7277345, upper bound: 106.7277387
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.7423469, upper bound: 106.7423502
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.7423469, upper bound: 106.7423501
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.7233875, upper bound: 106.7233828
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.7233858, upper bound: 106.7233857
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.7233875, upper bound: 106.7233850
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.7233872, upper bound: 106.7233857
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.7256768, upper bound: 106.7256764
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.7256768, upper bound: 106.7256764
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.7419713, upper bound: 106.7419693
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.7419716, upper bound: 106.7419693
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6668761, upper bound: 106.6668765
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6668760, upper bound: 106.6668769
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6398038, upper bound: 106.6398057
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.04
Output dim: 0, lower bound: -106.6398038, upper bound: 106.6398057

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6990113, upper bound: 106.6990142
time: 6.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6990113, upper bound: 106.6990143
time: 7.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7114789, upper bound: 106.7114774
time: 6.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7114789, upper bound: 106.7114774
time: 6.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 200

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6749068, upper bound: 106.6749068
time: 6.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6749068, upper bound: 106.6749068
time: 6.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6655235, upper bound: 106.6655235
time: 6.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6655235, upper bound: 106.6655235
time: 6.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6674573, upper bound: 106.6674544
time: 5.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6674569, upper bound: 106.6674555
time: 5.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6718065, upper bound: 106.6718061
time: 6.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6718065, upper bound: 106.6718063
time: 5.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 168

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6727330, upper bound: 106.6727320
time: 5.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6727331, upper bound: 106.6727321
time: 5.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 78

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6270272, upper bound: 106.6270263
time: 6.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6270272, upper bound: 106.6270263
time: 6.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6839307, upper bound: 106.6839274
time: 7.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6839309, upper bound: 106.6839285
time: 7.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -60.1304131, 47.8897781, -60.1304131, 47.8897781, -108.0201797, 108.0201797
1: -48.9951973, 41.5569344, -48.9951973, 41.5569344, -90.5521317, 90.5521317
2: -65.1370010, 42.1076736, -65.1370010, 42.1076736, -107.2446747, 107.2446747
3: -68.8342667, 36.2327805, -68.8342667, 36.2327805, -105.0670471, 105.0670471
4: -64.1016769, 48.5745659, -64.1016769, 48.5745659, -112.6762390, 112.6762390
5: -57.9381943, 45.3561211, -57.9381943, 45.3561211, -103.2943115, 103.2943115
6: -54.9252052, 52.6139221, -54.9252052, 52.6139221, -107.5391235, 107.5391235
7: -58.9155884, 50.4363098, -58.9155884, 50.4363098, -109.3518829, 109.3518829
8: -71.5229416, 47.6925201, -71.5229416, 47.6925201, -119.2154617, 119.2154617
9: -54.2685204, 52.9324722, -54.2685204, 52.9324722, -107.2009888, 107.2009888

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6826118, upper bound: 106.6826138
time: 6.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6826118, upper bound: 106.6826138
time: 6.36 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 13.73 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6990113, upper bound: 106.6990142
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6990113, upper bound: 106.6990143
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.7114789, upper bound: 106.7114774
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.7114789, upper bound: 106.7114774
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6749068, upper bound: 106.6749068
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6749068, upper bound: 106.6749068
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6655235, upper bound: 106.6655235
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6655235, upper bound: 106.6655235
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6674573, upper bound: 106.6674544
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6674569, upper bound: 106.6674555
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6718065, upper bound: 106.6718061
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6718065, upper bound: 106.6718063
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6727330, upper bound: 106.6727320
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6727331, upper bound: 106.6727321
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6270272, upper bound: 106.6270263
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6270272, upper bound: 106.6270263
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6839307, upper bound: 106.6839274
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6839309, upper bound: 106.6839285
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6826118, upper bound: 106.6826138
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 0, lower bound: -106.6826118, upper bound: 106.6826138
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.6931161, upper bound: 106.6931160
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.6931160, upper bound: 106.6931161
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.6790179, upper bound: 106.6790161
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.6790179, upper bound: 106.6790161
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.7277343, upper bound: 106.7277391
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.7277345, upper bound: 106.7277387
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.7423469, upper bound: 106.7423502
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.7423469, upper bound: 106.7423501
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.7233875, upper bound: 106.7233828
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.7233858, upper bound: 106.7233857
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.7233875, upper bound: 106.7233850
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.7233872, upper bound: 106.7233857
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.7256768, upper bound: 106.7256764
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.7256768, upper bound: 106.7256764
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.7419713, upper bound: 106.7419693
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.7419716, upper bound: 106.7419693
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.6668761, upper bound: 106.6668765
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 0, lower bound: -106.6668760, upper bound: 106.6668769

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 11.82 + 600.31 = 612.13 seconds
