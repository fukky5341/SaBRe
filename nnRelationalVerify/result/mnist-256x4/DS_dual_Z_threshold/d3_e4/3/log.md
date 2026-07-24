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
execution time: IAR + RelationalAnalysis = 0.78 + 10.91 = 11.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -106.7670618, upper bound: 106.7670618

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7633846, upper bound: 106.7633843
time: 7.72 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7633843, upper bound: 106.7633843
time: 6.73 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 14.51 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 14.51
Output dim: 0, lower bound: -106.7633846, upper bound: 106.7633843
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 14.51
Output dim: 0, lower bound: -106.7633843, upper bound: 106.7633843

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
time: 7.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
time: 7.51 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
time: 6.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
time: 6.98 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 14.27 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.27
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.27
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.27
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.27
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
time: 8.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
time: 8.33 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
time: 7.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
time: 8.42 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
time: 8.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
time: 8.27 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
time: 9.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
time: 8.72 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 18.59 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 18.59
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 18.59
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 18.59
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 18.59
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 18.59
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 18.59
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 18.59
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 18.59
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 6.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 6.32 seconds

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 9.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 8.61 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 6.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 6.62 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 8.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 7.87 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 7.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 7.16 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 7.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 7.54 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 7.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 7.42 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 7.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 6.47 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 14.43 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.43
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.43
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.43
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.43
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.43
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.43
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.43
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.43
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.43
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.43
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.43
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.43
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.43
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.43
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.43
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.43
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 6.72 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 6.81 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.14 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.11 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 6.94 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 6.68 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.38 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.68 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.91 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.62 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.08 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.18 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.90 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.90 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.79 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.63 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 13.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.53
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 5.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 5.62 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
time: 5.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
time: 5.91 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 5.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 6.04 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
time: 5.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
time: 5.91 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 6.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 6.16 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
time: 6.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
time: 6.22 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 6.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 6.20 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
time: 6.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
time: 6.40 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 15.58 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 15.58
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 15.58
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 15.58
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 15.58
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 15.58
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 15.58
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 15.58
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 15.58
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 15.58
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 15.58
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 15.58
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 15.58
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 15.58
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 15.58
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 15.58
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 15.58
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 11.70 + 593.16 = 604.86 seconds
