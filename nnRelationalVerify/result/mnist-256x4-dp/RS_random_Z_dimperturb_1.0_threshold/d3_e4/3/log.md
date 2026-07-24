## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 106.6602947382


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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
execution time: IAR + RelationalAnalysis = 1.46 + 11.31 = 12.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -106.7670618, upper bound: 106.7670618

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7599118, upper bound: 106.7599118
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7599118, upper bound: 106.7599118
time: 7.14 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.23 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.23
Output dim: 0, lower bound: -106.7599118, upper bound: 106.7599118
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.23
Output dim: 0, lower bound: -106.7599118, upper bound: 106.7599118

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7556052, upper bound: 106.7556052
time: 7.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7556052, upper bound: 106.7556052
time: 7.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7191331, upper bound: 106.7191331
time: 8.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7191331, upper bound: 106.7191331
time: 7.66 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 17.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.29
Output dim: 0, lower bound: -106.7556052, upper bound: 106.7556052
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.29
Output dim: 0, lower bound: -106.7556052, upper bound: 106.7556052
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.29
Output dim: 0, lower bound: -106.7191331, upper bound: 106.7191331
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.29
Output dim: 0, lower bound: -106.7191331, upper bound: 106.7191331

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7543889, upper bound: 106.7543889
time: 8.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7543889, upper bound: 106.7543889
time: 8.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 168

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7351095, upper bound: 106.7351095
time: 7.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7351095, upper bound: 106.7351095
time: 7.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7023989, upper bound: 106.7023989
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7023989, upper bound: 106.7023989
time: 6.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7143576, upper bound: 106.7143577
time: 7.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7143576, upper bound: 106.7143577
time: 7.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.74 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.74
Output dim: 0, lower bound: -106.7543889, upper bound: 106.7543889
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.74
Output dim: 0, lower bound: -106.7543889, upper bound: 106.7543889
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.74
Output dim: 0, lower bound: -106.7351095, upper bound: 106.7351095
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.74
Output dim: 0, lower bound: -106.7351095, upper bound: 106.7351095
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.74
Output dim: 0, lower bound: -106.7023989, upper bound: 106.7023989
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.74
Output dim: 0, lower bound: -106.7023989, upper bound: 106.7023989
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.74
Output dim: 0, lower bound: -106.7143576, upper bound: 106.7143577
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.74
Output dim: 0, lower bound: -106.7143576, upper bound: 106.7143577

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6625222, upper bound: 106.6625224
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6625222, upper bound: 106.6625224
time: 6.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7243772, upper bound: 106.7243772
time: 7.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7243772, upper bound: 106.7243772
time: 7.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7351095, upper bound: 106.7351069
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7351069, upper bound: 106.7351095
time: 6.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7141691, upper bound: 106.7141691
time: 7.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7141691, upper bound: 106.7141691
time: 7.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6989003, upper bound: 106.6989005
time: 8.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6989005, upper bound: 106.6989003
time: 6.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6992148, upper bound: 106.6992147
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6992148, upper bound: 106.6992147
time: 7.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6638861, upper bound: 106.6638858
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6638861, upper bound: 106.6638858
time: 6.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7003726, upper bound: 106.7003726
time: 8.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7003726, upper bound: 106.7003726
time: 8.58 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 20.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 0, lower bound: -106.6625222, upper bound: 106.6625224
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 0, lower bound: -106.6625222, upper bound: 106.6625224
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 0, lower bound: -106.7243772, upper bound: 106.7243772
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 0, lower bound: -106.7243772, upper bound: 106.7243772
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 0, lower bound: -106.7351095, upper bound: 106.7351069
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 0, lower bound: -106.7351069, upper bound: 106.7351095
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 0, lower bound: -106.7141691, upper bound: 106.7141691
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 0, lower bound: -106.7141691, upper bound: 106.7141691
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 0, lower bound: -106.6989003, upper bound: 106.6989005
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 0, lower bound: -106.6989005, upper bound: 106.6989003
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 0, lower bound: -106.6992148, upper bound: 106.6992147
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 0, lower bound: -106.6992148, upper bound: 106.6992147
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 0, lower bound: -106.6638861, upper bound: 106.6638858
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 0, lower bound: -106.6638861, upper bound: 106.6638858
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 0, lower bound: -106.7003726, upper bound: 106.7003726
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.67
Output dim: 0, lower bound: -106.7003726, upper bound: 106.7003726

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6569323, upper bound: 106.6569320
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6569323, upper bound: 106.6569322
time: 6.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 200

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6625216, upper bound: 106.6625224
time: 7.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6625224, upper bound: 106.6625215
time: 7.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7243772, upper bound: 106.7243772
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7243772, upper bound: 106.7243772
time: 7.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6386398, upper bound: 106.6386384
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6386398, upper bound: 106.6386384
time: 6.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 220

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7351095, upper bound: 106.7351069
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7351095, upper bound: 106.7351068
time: 6.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6763055, upper bound: 106.6762983
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6763055, upper bound: 106.6762983
time: 6.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6889856, upper bound: 106.6889857
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6889856, upper bound: 106.6889857
time: 6.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6889856, upper bound: 106.6889857
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6889856, upper bound: 106.6889857
time: 6.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6690064, upper bound: 106.6690068
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6690064, upper bound: 106.6690068
time: 7.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6793758, upper bound: 106.6793758
time: 6.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6793758, upper bound: 106.6793758
time: 7.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6992125, upper bound: 106.6992147
time: 7.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6992148, upper bound: 106.6992121
time: 6.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6400912, upper bound: 106.6400912
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6400912, upper bound: 106.6400912
time: 6.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6600705, upper bound: 106.6600705
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6600705, upper bound: 106.6600705
time: 6.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6573394, upper bound: 106.6573394
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6573394, upper bound: 106.6573395
time: 6.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6950302, upper bound: 106.6950302
time: 6.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6950302, upper bound: 106.6950302
time: 6.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6493071, upper bound: 106.6493071
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6493071, upper bound: 106.6493071
time: 6.21 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 15.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6569323, upper bound: 106.6569320
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6569323, upper bound: 106.6569322
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6625216, upper bound: 106.6625224
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6625224, upper bound: 106.6625215
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.7243772, upper bound: 106.7243772
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.7243772, upper bound: 106.7243772
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6386398, upper bound: 106.6386384
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6386398, upper bound: 106.6386384
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.7351095, upper bound: 106.7351069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.7351095, upper bound: 106.7351068
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6763055, upper bound: 106.6762983
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6763055, upper bound: 106.6762983
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6889856, upper bound: 106.6889857
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6889856, upper bound: 106.6889857
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6889856, upper bound: 106.6889857
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6889856, upper bound: 106.6889857
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6690064, upper bound: 106.6690068
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6690064, upper bound: 106.6690068
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6793758, upper bound: 106.6793758
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6793758, upper bound: 106.6793758
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6992125, upper bound: 106.6992147
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6992148, upper bound: 106.6992121
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6400912, upper bound: 106.6400912
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6400912, upper bound: 106.6400912
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6600705, upper bound: 106.6600705
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6600705, upper bound: 106.6600705
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6573394, upper bound: 106.6573394
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6573394, upper bound: 106.6573395
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6950302, upper bound: 106.6950302
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6950302, upper bound: 106.6950302
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6493071, upper bound: 106.6493071
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.92
Output dim: 0, lower bound: -106.6493071, upper bound: 106.6493071

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.5964951, upper bound: 106.5964949
time: 6.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.5964951, upper bound: 106.5964949
time: 6.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6484426, upper bound: 106.6484440
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6484426, upper bound: 106.6484440
time: 6.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7194114, upper bound: 106.7194152
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7194147, upper bound: 106.7194120
time: 6.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 168

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 201

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7243771, upper bound: 106.7243772
time: 7.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7243771, upper bound: 106.7243772
time: 7.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7088293, upper bound: 106.7088279
time: 8.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7088293, upper bound: 106.7088279
time: 8.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=64, inp2_unstable=64, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=151, inp2_unstable=151, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=223, inp2_unstable=223, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7298478, upper bound: 106.7298492
time: 6.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7298475, upper bound: 106.7298512
time: 7.57 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 15.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.78
Output dim: 0, lower bound: -106.5964951, upper bound: 106.5964949
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.78
Output dim: 0, lower bound: -106.5964951, upper bound: 106.5964949
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 15.78
Output dim: 0, lower bound: -106.6484426, upper bound: 106.6484440
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 15.78
Output dim: 0, lower bound: -106.6484426, upper bound: 106.6484440
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.78
Output dim: 0, lower bound: -106.7194114, upper bound: 106.7194152
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.78
Output dim: 0, lower bound: -106.7194147, upper bound: 106.7194120
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.78
Output dim: 0, lower bound: -106.7243771, upper bound: 106.7243772
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.78
Output dim: 0, lower bound: -106.7243771, upper bound: 106.7243772
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.78
Output dim: 0, lower bound: -106.7088293, upper bound: 106.7088279
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.78
Output dim: 0, lower bound: -106.7088293, upper bound: 106.7088279
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.78
Output dim: 0, lower bound: -106.7298478, upper bound: 106.7298492
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.78
Output dim: 0, lower bound: -106.7298475, upper bound: 106.7298512
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 0, lower bound: -106.6763055, upper bound: 106.6762983
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 0, lower bound: -106.6763055, upper bound: 106.6762983
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 0, lower bound: -106.6889856, upper bound: 106.6889857
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 0, lower bound: -106.6889856, upper bound: 106.6889857
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 0, lower bound: -106.6889856, upper bound: 106.6889857
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 0, lower bound: -106.6889856, upper bound: 106.6889857
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 0, lower bound: -106.6690064, upper bound: 106.6690068
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 0, lower bound: -106.6690064, upper bound: 106.6690068
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 0, lower bound: -106.6793758, upper bound: 106.6793758
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 0, lower bound: -106.6793758, upper bound: 106.6793758
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 0, lower bound: -106.6992125, upper bound: 106.6992147
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 0, lower bound: -106.6992148, upper bound: 106.6992121
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 0, lower bound: -106.6950302, upper bound: 106.6950302
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 0, lower bound: -106.6950302, upper bound: 106.6950302

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 12.77 + 597.19 = 609.96 seconds
