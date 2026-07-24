## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 106.6602947382
Search space: {k/256 | k = 1, 2, ..., 12}


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

## BASE Result
execution time: IAR + LP analysis = 1.28 + 9.21 = 10.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -106.7671438, upper bound: 106.7671437


# Binary Search by BASE starts (time budget: 1989.51 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=108.02017974853516
rel_dist={0: [-106.7671031285499, 106.7671031285499]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=108.02017974853516
rel_dist={0: [-106.76706178863964, 106.76706178863964]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=108.02017974853516
rel_dist={0: [-106.76692036843552, 106.76692036843548]}

## Binary Search Result
Binary search time: 38.22 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1951.30 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7634050, upper bound: 106.7634042
time: 7.96 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7634044, upper bound: 106.7634050
time: 8.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.44 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.44
Output dim: 0, lower bound: -106.7634050, upper bound: 106.7634042
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.44
Output dim: 0, lower bound: -106.7634044, upper bound: 106.7634050

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

Time for backsubstitution: 1.20 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435559, upper bound: 106.7435559
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435559, upper bound: 106.7435559
time: 6.56 seconds

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

Time for backsubstitution: 1.19 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435559, upper bound: 106.7435559
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435559, upper bound: 106.7435559
time: 6.56 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.67 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.67
Output dim: 0, lower bound: -106.7435559, upper bound: 106.7435559
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.67
Output dim: 0, lower bound: -106.7435559, upper bound: 106.7435559
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.67
Output dim: 0, lower bound: -106.7435559, upper bound: 106.7435559
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.67
Output dim: 0, lower bound: -106.7435559, upper bound: 106.7435559

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

Time for backsubstitution: 1.22 seconds

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
Output dim: 0, lower bound: -106.7319806, upper bound: 106.7319790
time: 7.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319806, upper bound: 106.7319790
time: 8.20 seconds

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

Time for backsubstitution: 1.19 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319806, upper bound: 106.7319790
time: 8.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319806, upper bound: 106.7319790
time: 7.86 seconds

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

Time for backsubstitution: 1.21 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319790, upper bound: 106.7319806
time: 8.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319790, upper bound: 106.7319806
time: 8.27 seconds

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

Time for backsubstitution: 1.21 seconds

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
Output dim: 0, lower bound: -106.7319790, upper bound: 106.7319806
time: 8.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319790, upper bound: 106.7319806
time: 7.88 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 17.61 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.61
Output dim: 0, lower bound: -106.7319806, upper bound: 106.7319790
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.61
Output dim: 0, lower bound: -106.7319806, upper bound: 106.7319790
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.61
Output dim: 0, lower bound: -106.7319806, upper bound: 106.7319790
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.61
Output dim: 0, lower bound: -106.7319806, upper bound: 106.7319790
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.61
Output dim: 0, lower bound: -106.7319790, upper bound: 106.7319806
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.61
Output dim: 0, lower bound: -106.7319790, upper bound: 106.7319806
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.61
Output dim: 0, lower bound: -106.7319790, upper bound: 106.7319806
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.61
Output dim: 0, lower bound: -106.7319790, upper bound: 106.7319806

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

Time for backsubstitution: 1.32 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910267, upper bound: 106.6910255
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910267, upper bound: 106.6910255
time: 5.92 seconds

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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910267, upper bound: 106.6910255
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910267, upper bound: 106.6910255
time: 5.94 seconds

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

Time for backsubstitution: 1.24 seconds

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
Output dim: 0, lower bound: -106.6910267, upper bound: 106.6910255
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910267, upper bound: 106.6910255
time: 6.03 seconds

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

Time for backsubstitution: 1.22 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910267, upper bound: 106.6910255
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910267, upper bound: 106.6910255
time: 6.10 seconds

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

Time for backsubstitution: 1.18 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910255, upper bound: 106.6910267
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910255, upper bound: 106.6910267
time: 5.75 seconds

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

Time for backsubstitution: 1.20 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910255, upper bound: 106.6910268
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910255, upper bound: 106.6910268
time: 5.23 seconds

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

Time for backsubstitution: 1.23 seconds

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
Output dim: 0, lower bound: -106.6910255, upper bound: 106.6910268
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910255, upper bound: 106.6910268
time: 4.98 seconds

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

Time for backsubstitution: 1.21 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910255, upper bound: 106.6910267
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910255, upper bound: 106.6910267
time: 5.22 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 11.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.83
Output dim: 0, lower bound: -106.6910267, upper bound: 106.6910255
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.83
Output dim: 0, lower bound: -106.6910267, upper bound: 106.6910255
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.83
Output dim: 0, lower bound: -106.6910267, upper bound: 106.6910255
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.83
Output dim: 0, lower bound: -106.6910267, upper bound: 106.6910255
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.83
Output dim: 0, lower bound: -106.6910267, upper bound: 106.6910255
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.83
Output dim: 0, lower bound: -106.6910267, upper bound: 106.6910255
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.83
Output dim: 0, lower bound: -106.6910267, upper bound: 106.6910255
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.83
Output dim: 0, lower bound: -106.6910267, upper bound: 106.6910255
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.83
Output dim: 0, lower bound: -106.6910255, upper bound: 106.6910267
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.83
Output dim: 0, lower bound: -106.6910255, upper bound: 106.6910267
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.83
Output dim: 0, lower bound: -106.6910255, upper bound: 106.6910268
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.83
Output dim: 0, lower bound: -106.6910255, upper bound: 106.6910268
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.83
Output dim: 0, lower bound: -106.6910255, upper bound: 106.6910268
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.83
Output dim: 0, lower bound: -106.6910255, upper bound: 106.6910268
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.83
Output dim: 0, lower bound: -106.6910255, upper bound: 106.6910267
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.83
Output dim: 0, lower bound: -106.6910255, upper bound: 106.6910267

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

Time for backsubstitution: 1.22 seconds

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
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
time: 6.81 seconds

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

Time for backsubstitution: 1.21 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
time: 6.77 seconds

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

Time for backsubstitution: 1.19 seconds

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
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869653
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
time: 6.62 seconds

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

Time for backsubstitution: 1.20 seconds

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
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869653
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
time: 6.68 seconds

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

Time for backsubstitution: 1.21 seconds

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
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
time: 7.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
time: 6.78 seconds

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

Time for backsubstitution: 1.20 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
time: 6.84 seconds

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

Time for backsubstitution: 1.23 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869653
time: 6.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
time: 6.76 seconds

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

Time for backsubstitution: 1.21 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869653
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
time: 6.74 seconds

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

Time for backsubstitution: 1.23 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869654
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
time: 7.74 seconds

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

Time for backsubstitution: 1.20 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869654
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
time: 7.76 seconds

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

Time for backsubstitution: 1.20 seconds

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
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869653
time: 6.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869653
time: 7.97 seconds

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

Time for backsubstitution: 1.21 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869653
time: 6.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869653
time: 8.05 seconds

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

Time for backsubstitution: 1.22 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869654
time: 6.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
time: 7.96 seconds

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

Time for backsubstitution: 1.23 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869654
time: 7.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
time: 8.00 seconds

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

Time for backsubstitution: 1.22 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869653
time: 6.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
time: 7.53 seconds

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

Time for backsubstitution: 1.24 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869653
time: 6.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
time: 7.82 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 15.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869653
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869653
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869653
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869653
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869654
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869654
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869653
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869653
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869653
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869653
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869654
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869654
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869653
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869653
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.96
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654

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

Time for backsubstitution: 1.23 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016447
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016447
time: 5.44 seconds

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

Time for backsubstitution: 1.23 seconds

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
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016451
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016451
time: 6.01 seconds

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

Time for backsubstitution: 1.23 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016447
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016447
time: 5.41 seconds

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

Time for backsubstitution: 1.21 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016451
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016451
time: 6.22 seconds

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

Time for backsubstitution: 1.22 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016447
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016447
time: 4.99 seconds

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

Time for backsubstitution: 1.22 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016451
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016451
time: 6.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.50 seconds

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

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016447
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016447
time: 5.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.53 seconds

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

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016451
time: 6.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016451
time: 6.52 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 17.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 17.53
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016447
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 17.53
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016447
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 17.53
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016451
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 17.53
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016451
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 17.53
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016447
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 17.53
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016447
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 17.53
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016451
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 17.53
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016451
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 17.53
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016447
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 17.53
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016447
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 17.53
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016451
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 17.53
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016451
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 17.53
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 17.53
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016447
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 17.53
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016451
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 17.53
Output dim: 0, lower bound: -106.6016447, upper bound: 106.6016451
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869653
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869653
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869652
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869654
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869654
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869653
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869653
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869653
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869653
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869654
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869654
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869653
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869652, upper bound: 106.6869653
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.53
Output dim: 0, lower bound: -106.6869653, upper bound: 106.6869654
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=108.02017974853516
rel_dist={0: [-106.7671031285499, 106.7671031285499]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

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
time: 8.41 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7633843, upper bound: 106.7633843
time: 7.40 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.97 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.97
Output dim: 0, lower bound: -106.7633846, upper bound: 106.7633843
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.97
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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
time: 7.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
time: 8.15 seconds

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

Time for backsubstitution: 1.34 seconds

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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
time: 7.39 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.77 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.77
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.77
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.77
Output dim: 0, lower bound: -106.7435498, upper bound: 106.7435498
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.77
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

Time for backsubstitution: 1.27 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
time: 9.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
time: 8.92 seconds

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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
time: 9.05 seconds

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

Time for backsubstitution: 1.31 seconds

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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
time: 8.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
time: 8.79 seconds

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

Time for backsubstitution: 1.29 seconds

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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
time: 9.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
time: 8.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 19.51 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.51
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.51
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.51
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.51
Output dim: 0, lower bound: -106.7319687, upper bound: 106.7319681
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.51
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.51
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 19.51
Output dim: 0, lower bound: -106.7319681, upper bound: 106.7319687
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 19.51
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

Time for backsubstitution: 1.26 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 6.51 seconds

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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 9.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 9.85 seconds

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

Time for backsubstitution: 1.26 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 7.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 7.23 seconds

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

Time for backsubstitution: 1.34 seconds

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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 8.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
time: 8.48 seconds

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

Time for backsubstitution: 1.33 seconds

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
time: 7.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 7.65 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 8.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 8.15 seconds

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

Time for backsubstitution: 1.34 seconds

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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 8.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 7.97 seconds

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

Time for backsubstitution: 1.27 seconds

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
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
time: 6.79 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.67
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.67
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.67
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.67
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.67
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.67
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.67
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.67
Output dim: 0, lower bound: -106.6910194, upper bound: 106.6910190
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.67
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.67
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.67
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.67
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.67
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.67
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.67
Output dim: 0, lower bound: -106.6910189, upper bound: 106.6910194
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.67
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

Time for backsubstitution: 1.19 seconds

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
time: 7.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.07 seconds

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

Time for backsubstitution: 1.19 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.06 seconds

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

Time for backsubstitution: 1.19 seconds

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
time: 6.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.23 seconds

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

Time for backsubstitution: 1.28 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.34 seconds

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

Time for backsubstitution: 1.33 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.12 seconds

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

Time for backsubstitution: 1.19 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.03 seconds

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

Time for backsubstitution: 1.26 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.82 seconds

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

Time for backsubstitution: 1.16 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
time: 7.72 seconds

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

Time for backsubstitution: 1.21 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 8.47 seconds

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

Time for backsubstitution: 1.31 seconds

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
time: 7.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 8.24 seconds

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

Time for backsubstitution: 1.32 seconds

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
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.60 seconds

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

Time for backsubstitution: 1.33 seconds

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
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.77 seconds

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

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.29 seconds

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

Time for backsubstitution: 1.31 seconds

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
time: 7.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.28 seconds

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

Time for backsubstitution: 1.24 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 7.21 seconds

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

Time for backsubstitution: 1.24 seconds

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
time: 6.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
time: 6.91 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869578, upper bound: 106.6869577
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -106.6869577, upper bound: 106.6869578
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
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

Time for backsubstitution: 1.25 seconds

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

Time for candidate selection: 0.12 seconds

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
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 5.83 seconds

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

Time for backsubstitution: 1.24 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
time: 6.40 seconds

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

Time for backsubstitution: 1.23 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016248
time: 6.54 seconds

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

Time for backsubstitution: 1.26 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -106.6016248, upper bound: 106.6016251
time: 6.45 seconds

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

Time for backsubstitution: 1.21 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=108.02017974853516
rel_dist={0: [-106.76706178863964, 106.76706178863964]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7633370, upper bound: 106.7633371
time: 9.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7633371, upper bound: 106.7633370
time: 8.58 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 18.01 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 18.01
Output dim: 0, lower bound: -106.7633370, upper bound: 106.7633371
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 18.01
Output dim: 0, lower bound: -106.7633371, upper bound: 106.7633370

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

Time for backsubstitution: 1.24 seconds

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
Output dim: 0, lower bound: -106.7434330, upper bound: 106.7434330
time: 9.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7434330, upper bound: 106.7434330
time: 10.31 seconds

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

Time for backsubstitution: 1.27 seconds

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
Output dim: 0, lower bound: -106.7434330, upper bound: 106.7434330
time: 10.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7434330, upper bound: 106.7434330
time: 8.72 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 20.45 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.45
Output dim: 0, lower bound: -106.7434330, upper bound: 106.7434330
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.45
Output dim: 0, lower bound: -106.7434330, upper bound: 106.7434330
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.45
Output dim: 0, lower bound: -106.7434330, upper bound: 106.7434330
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.45
Output dim: 0, lower bound: -106.7434330, upper bound: 106.7434330

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

Time for backsubstitution: 1.24 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319330, upper bound: 106.7319327
time: 10.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319327, upper bound: 106.7319327
time: 9.09 seconds

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

Time for backsubstitution: 1.23 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319330, upper bound: 106.7319327
time: 9.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319327, upper bound: 106.7319327
time: 8.83 seconds

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

Time for backsubstitution: 1.21 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319327, upper bound: 106.7319330
time: 10.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319327, upper bound: 106.7319330
time: 11.17 seconds

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

Time for backsubstitution: 1.22 seconds

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
Output dim: 0, lower bound: -106.7319327, upper bound: 106.7319330
time: 8.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.7319327, upper bound: 106.7319330
time: 8.70 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.87 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.87
Output dim: 0, lower bound: -106.7319330, upper bound: 106.7319327
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.87
Output dim: 0, lower bound: -106.7319327, upper bound: 106.7319327
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.87
Output dim: 0, lower bound: -106.7319330, upper bound: 106.7319327
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.87
Output dim: 0, lower bound: -106.7319327, upper bound: 106.7319327
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.87
Output dim: 0, lower bound: -106.7319327, upper bound: 106.7319330
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.87
Output dim: 0, lower bound: -106.7319327, upper bound: 106.7319330
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.87
Output dim: 0, lower bound: -106.7319327, upper bound: 106.7319330
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.87
Output dim: 0, lower bound: -106.7319327, upper bound: 106.7319330

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

Time for backsubstitution: 1.21 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910022, upper bound: 106.6910021
time: 6.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910022, upper bound: 106.6910021
time: 6.96 seconds

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

Time for backsubstitution: 1.23 seconds

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
Output dim: 0, lower bound: -106.6910022, upper bound: 106.6910021
time: 8.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910022, upper bound: 106.6910021
time: 8.13 seconds

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

Time for backsubstitution: 1.24 seconds

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
Output dim: 0, lower bound: -106.6910023, upper bound: 106.6910021
time: 9.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910021, upper bound: 106.6910021
time: 9.01 seconds

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

Time for backsubstitution: 1.23 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910022, upper bound: 106.6910021
time: 8.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910022, upper bound: 106.6910021
time: 8.45 seconds

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

Time for backsubstitution: 1.21 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
time: 7.21 seconds

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

Time for backsubstitution: 1.24 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
time: 8.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
time: 7.95 seconds

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

Time for backsubstitution: 1.23 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
time: 8.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
time: 9.64 seconds

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

Time for backsubstitution: 1.24 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
time: 8.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
time: 8.66 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 18.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.6910022, upper bound: 106.6910021
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.6910022, upper bound: 106.6910021
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.6910022, upper bound: 106.6910021
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.6910022, upper bound: 106.6910021
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.6910023, upper bound: 106.6910021
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.6910021, upper bound: 106.6910021
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.6910022, upper bound: 106.6910021
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.6910022, upper bound: 106.6910021
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.17
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021

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

Time for backsubstitution: 1.22 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869511, upper bound: 106.6869511
time: 10.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869511
time: 8.83 seconds

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

Time for backsubstitution: 1.22 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869511, upper bound: 106.6869511
time: 10.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869511
time: 8.42 seconds

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

Time for backsubstitution: 1.22 seconds

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
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869512
time: 10.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869511
time: 7.18 seconds

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

Time for backsubstitution: 1.23 seconds

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
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869512
time: 9.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869511
time: 7.21 seconds

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

Time for backsubstitution: 1.22 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869512
time: 9.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869511
time: 9.10 seconds

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

Time for backsubstitution: 1.23 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869511, upper bound: 106.6869511
time: 135.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869511
time: 8.07 seconds

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

Time for backsubstitution: 1.26 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869512
time: 9.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869511
time: 7.65 seconds

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

Time for backsubstitution: 1.20 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869512
time: 9.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869511
time: 6.94 seconds

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

Time for backsubstitution: 1.21 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869512
time: 9.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869511, upper bound: 106.6869512
time: 8.17 seconds

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

Time for backsubstitution: 1.22 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869512
time: 9.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -106.6869511, upper bound: 106.6869512
time: 8.20 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 19.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869511, upper bound: 106.6869511
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869511
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869511, upper bound: 106.6869511
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869511
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869512
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869511
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869512
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869511
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869512
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869511
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869511, upper bound: 106.6869511
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869511
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869512
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869511
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869512
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869511
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869512
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869511, upper bound: 106.6869512
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869512, upper bound: 106.6869512
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.20
Output dim: 0, lower bound: -106.6869511, upper bound: 106.6869512
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.20
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.20
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.20
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.20
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.20
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.20
Output dim: 0, lower bound: -106.6910020, upper bound: 106.6910021
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=108.02017974853516
rel_dist={0: [-106.76692036843552, 106.76692036843548]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1817.40 seconds
