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
execution time: IAR + RelationalAnalysis = 1.49 + 11.31 = 12.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -106.7670618, upper bound: 106.7670618

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7633846, upper bound: 106.7633843
time: 8.09 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7633843, upper bound: 106.7633843
time: 7.04 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.29
Output dim: 0, lower bound: -106.7633846, upper bound: 106.7633843
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.29
Output dim: 0, lower bound: -106.7633843, upper bound: 106.7633843

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
time: 7.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
time: 7.69 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
time: 6.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
time: 7.11 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.16 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.16
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.16
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.16
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.16
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
time: 9.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
time: 8.63 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
time: 8.72 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
time: 8.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
time: 8.60 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
time: 8.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
time: 8.45 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.87 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.87
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.87
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.87
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.87
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.87
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.87
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.87
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.87
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687

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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 6.20 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 9.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 8.87 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 6.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 6.86 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 8.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 8.03 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 7.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 7.30 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 7.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 7.75 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 7.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 7.65 seconds

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
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 6.68 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.54 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.00 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 6.97 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.11 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.14 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 6.88 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 6.88 seconds

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.63 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.63 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 8.09 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.88 seconds

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.39 seconds

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.34 seconds

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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.08 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.19 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.00 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.77 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.59
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 5.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
time: 6.32 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 6.30 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
time: 6.27 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 6.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 6.53 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 201
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 200
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 12.80 + 588.86 = 601.66 seconds
