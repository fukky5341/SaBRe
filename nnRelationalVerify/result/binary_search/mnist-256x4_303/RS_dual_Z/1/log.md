## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 70.0244797244
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782)
1: (-35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156)
2: (-45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539)
3: (-48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106)
4: (-45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758)
5: (-39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896)
6: (-36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245)
7: (-40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293)
8: (-54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865)
9: (-35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065)

## BASE Result
execution time: IAR + LP analysis = 1.23 + 8.82 = 10.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -70.0817842, upper bound: 70.0817842


# Binary Search by BASE starts (time budget: 2689.95 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=82.72178649902344
rel_dist={8: [-70.0816870801838, 70.08168706258579]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=82.72178649902344
rel_dist={8: [-70.08157121987698, 70.08157121987699]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=82.72178649902344
rel_dist={8: [-70.08140925617062, 70.08140925618659]}

## Binary Search Result
Binary search time: 41.14 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2648.81 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0528356, upper bound: 70.0528356
time: 6.83 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0528356, upper bound: 70.0528356
time: 7.21 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.17 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.17
Output dim: 8, lower bound: -70.0528356, upper bound: 70.0528356
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.17
Output dim: 8, lower bound: -70.0528356, upper bound: 70.0528356

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651
time: 6.41 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651
time: 6.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651
time: 6.39 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.10
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.10
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.10
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.10
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
time: 7.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
time: 6.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
time: 7.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
time: 7.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
time: 6.75 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.33 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.33
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.33
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.33
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.33
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.33
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.33
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.33
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.33
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
time: 6.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
time: 5.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
time: 5.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
time: 6.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
time: 5.26 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 12.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.88
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.88
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.88
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.88
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.88
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.88
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.88
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.88
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.88
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.88
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.88
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.88
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.88
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.88
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.88
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.88
Output dim: 8, lower bound: -70.0244509, upper bound: 70.0244510
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=82.72178649902344
rel_dist={8: [-70.0816870801838, 70.08168706258579]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0528648, upper bound: 70.0528648
time: 7.00 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0528648, upper bound: 70.0528648
time: 6.99 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.13
Output dim: 8, lower bound: -70.0528648, upper bound: 70.0528648
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.13
Output dim: 8, lower bound: -70.0528648, upper bound: 70.0528648

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369696, upper bound: 70.0369696
time: 11.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369696, upper bound: 70.0369696
time: 5.01 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369696, upper bound: 70.0369696
time: 10.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369696, upper bound: 70.0369696
time: 5.01 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.48 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.48
Output dim: 8, lower bound: -70.0369696, upper bound: 70.0369696
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.48
Output dim: 8, lower bound: -70.0369696, upper bound: 70.0369696
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.48
Output dim: 8, lower bound: -70.0369696, upper bound: 70.0369696
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.48
Output dim: 8, lower bound: -70.0369696, upper bound: 70.0369696

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328917, upper bound: 70.0328917
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328917, upper bound: 70.0328917
time: 5.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328917, upper bound: 70.0328917
time: 7.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328917, upper bound: 70.0328917
time: 7.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328917, upper bound: 70.0328917
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328917, upper bound: 70.0328917
time: 6.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328917, upper bound: 70.0328917
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328917, upper bound: 70.0328917
time: 7.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.94 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 8, lower bound: -70.0328917, upper bound: 70.0328917
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 8, lower bound: -70.0328917, upper bound: 70.0328917
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 8, lower bound: -70.0328917, upper bound: 70.0328917
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 8, lower bound: -70.0328917, upper bound: 70.0328917
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 8, lower bound: -70.0328917, upper bound: 70.0328917
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 8, lower bound: -70.0328917, upper bound: 70.0328917
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 8, lower bound: -70.0328917, upper bound: 70.0328917
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 8, lower bound: -70.0328917, upper bound: 70.0328917

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
time: 7.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
time: 6.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
time: 5.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
time: 5.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
time: 7.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
time: 6.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
time: 5.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
time: 5.45 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 12.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.30
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.30
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.30
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.30
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.30
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.30
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.30
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.30
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.30
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.30
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.30
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.30
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.30
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.30
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.30
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.30
Output dim: 8, lower bound: -70.0244722, upper bound: 70.0244722
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=82.72178649902344
rel_dist={8: [-70.08174123959981, 70.08174122023962]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0528815, upper bound: 70.0528815
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0528815, upper bound: 70.0528815
time: 6.25 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.63 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.63
Output dim: 8, lower bound: -70.0528815, upper bound: 70.0528815
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.63
Output dim: 8, lower bound: -70.0528815, upper bound: 70.0528815

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369722, upper bound: 70.0369722
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369722, upper bound: 70.0369722
time: 5.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369722, upper bound: 70.0369722
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369722, upper bound: 70.0369722
time: 5.27 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 12.36 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.36
Output dim: 8, lower bound: -70.0369722, upper bound: 70.0369722
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.36
Output dim: 8, lower bound: -70.0369722, upper bound: 70.0369722
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.36
Output dim: 8, lower bound: -70.0369722, upper bound: 70.0369722
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.36
Output dim: 8, lower bound: -70.0369722, upper bound: 70.0369722

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0329025, upper bound: 70.0329025
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0329025, upper bound: 70.0329025
time: 4.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0329025, upper bound: 70.0329025
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0329025, upper bound: 70.0329025
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0329025, upper bound: 70.0329025
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0329025, upper bound: 70.0329025
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0329025, upper bound: 70.0329025
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0329025, upper bound: 70.0329025
time: 5.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 10.94 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 10.94
Output dim: 8, lower bound: -70.0329025, upper bound: 70.0329025
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 10.94
Output dim: 8, lower bound: -70.0329025, upper bound: 70.0329025
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 10.94
Output dim: 8, lower bound: -70.0329025, upper bound: 70.0329025
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 10.94
Output dim: 8, lower bound: -70.0329025, upper bound: 70.0329025
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 10.94
Output dim: 8, lower bound: -70.0329025, upper bound: 70.0329025
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 10.94
Output dim: 8, lower bound: -70.0329025, upper bound: 70.0329025
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 10.94
Output dim: 8, lower bound: -70.0329025, upper bound: 70.0329025
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 10.94
Output dim: 8, lower bound: -70.0329025, upper bound: 70.0329025

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
time: 5.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
time: 6.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
time: 6.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
time: 5.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
time: 5.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
time: 5.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
time: 6.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
time: 6.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
time: 6.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
time: 6.01 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.60
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.60
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.60
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.60
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.60
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.60
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.60
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.60
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.60
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.60
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.60
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.60
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.60
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.60
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.60
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.60
Output dim: 8, lower bound: -70.0244828, upper bound: 70.0244828

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
time: 6.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
time: 6.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
time: 6.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
time: 6.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
time: 6.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
time: 6.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
time: 7.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
time: 7.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
time: 6.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
time: 6.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
time: 6.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
time: 6.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
time: 7.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
time: 6.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
time: 6.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
time: 7.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
time: 7.14 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214966, upper bound: 70.0214903
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.63
Output dim: 8, lower bound: -70.0214903, upper bound: 70.0214966
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=82.72178649902344
rel_dist={8: [-70.08177041452497, 70.08177039184733]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0528897, upper bound: 70.0528897
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0528897, upper bound: 70.0528897
time: 4.63 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.38
Output dim: 8, lower bound: -70.0528897, upper bound: 70.0528897
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.38
Output dim: 8, lower bound: -70.0528897, upper bound: 70.0528897

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369735, upper bound: 70.0369735
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369735, upper bound: 70.0369735
time: 6.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369735, upper bound: 70.0369735
time: 6.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369735, upper bound: 70.0369735
time: 6.50 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.36 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.36
Output dim: 8, lower bound: -70.0369735, upper bound: 70.0369735
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.36
Output dim: 8, lower bound: -70.0369735, upper bound: 70.0369735
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.36
Output dim: 8, lower bound: -70.0369735, upper bound: 70.0369735
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.36
Output dim: 8, lower bound: -70.0369735, upper bound: 70.0369735

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0329075, upper bound: 70.0329075
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0329075, upper bound: 70.0329075
time: 5.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0329075, upper bound: 70.0329075
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0329075, upper bound: 70.0329075
time: 5.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0329075, upper bound: 70.0329075
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0329075, upper bound: 70.0329075
time: 5.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0329075, upper bound: 70.0329075
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0329075, upper bound: 70.0329075
time: 5.20 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 12.41 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.41
Output dim: 8, lower bound: -70.0329075, upper bound: 70.0329075
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.41
Output dim: 8, lower bound: -70.0329075, upper bound: 70.0329075
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.41
Output dim: 8, lower bound: -70.0329075, upper bound: 70.0329075
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.41
Output dim: 8, lower bound: -70.0329075, upper bound: 70.0329075
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.41
Output dim: 8, lower bound: -70.0329075, upper bound: 70.0329075
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.41
Output dim: 8, lower bound: -70.0329075, upper bound: 70.0329075
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.41
Output dim: 8, lower bound: -70.0329075, upper bound: 70.0329075
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.41
Output dim: 8, lower bound: -70.0329075, upper bound: 70.0329075

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
time: 4.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
time: 10.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
time: 5.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
time: 6.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
time: 5.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
time: 4.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
time: 9.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
time: 5.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
time: 6.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
time: 5.82 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 12.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.29
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.29
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.29
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.29
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.29
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.29
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.29
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.29
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.29
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.29
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.29
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.29
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.29
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.29
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.29
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.29
Output dim: 8, lower bound: -70.0244882, upper bound: 70.0244882

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
time: 6.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
time: 8.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
time: 7.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
time: 7.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
time: 12.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
time: 7.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
time: 18.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
time: 7.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
time: 6.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
time: 12.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
time: 7.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
time: 13.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
time: 6.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
time: 6.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
time: 8.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
time: 7.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
time: 11.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
time: 7.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
time: 18.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
time: 7.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
time: 6.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
time: 12.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
time: 7.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -40.2857857, 32.0787888, -40.2857857, 32.0787888, -72.3645706, 72.3645782
1: -35.5273781, 28.5705376, -35.5273781, 28.5705376, -64.0979080, 64.0979156
2: -45.4768066, 28.2778454, -45.4768066, 28.2778454, -73.7546539, 73.7546539
3: -48.9736557, 24.1043568, -48.9736557, 24.1043568, -73.0780106, 73.0780106
4: -45.8082924, 32.8718872, -45.8082924, 32.8718872, -78.6801758, 78.6801758
5: -39.8284340, 30.9265594, -39.8284340, 30.9265594, -70.7549896, 70.7549896
6: -36.8836212, 36.2383995, -36.8836212, 36.2383995, -73.1220245, 73.1220245
7: -40.5858040, 37.3845253, -40.5858040, 37.3845253, -77.9703293, 77.9703293
8: -54.7539024, 27.9678822, -54.7539024, 27.9678822, -82.7217865, 82.7217865
9: -35.9190674, 36.0372391, -35.9190674, 36.0372391, -71.9563065, 71.9563065

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
time: 14.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
time: 6.61 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214998, upper bound: 70.0214924
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.85
Output dim: 8, lower bound: -70.0214924, upper bound: 70.0214998
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=82.72178649902344
rel_dist={8: [-70.08178419122864, 70.08178416425324]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 1456.11 seconds
