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
execution time: IAR + LP analysis = 1.26 + 8.92 = 10.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -70.0817842, upper bound: 70.0817842


# Binary Search by BASE starts (time budget: 2689.82 seconds, max iter: 100)

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
Binary search time: 41.62 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2648.20 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0628157, upper bound: 70.0628157
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0628158, upper bound: 70.0628158
time: 7.99 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.12 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.12
Output dim: 8, lower bound: -70.0628157, upper bound: 70.0628157
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.12
Output dim: 8, lower bound: -70.0628158, upper bound: 70.0628158

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
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0628157, upper bound: 70.0628158
time: 7.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0628157, upper bound: 70.0628157
time: 8.15 seconds

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
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0400175, upper bound: 70.0400175
time: 7.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0400175, upper bound: 70.0400175
time: 19.19 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 27.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.55
Output dim: 8, lower bound: -70.0628157, upper bound: 70.0628158
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.55
Output dim: 8, lower bound: -70.0628157, upper bound: 70.0628157
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.55
Output dim: 8, lower bound: -70.0400175, upper bound: 70.0400175
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.55
Output dim: 8, lower bound: -70.0400175, upper bound: 70.0400175

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
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0592913, upper bound: 70.0592928
time: 9.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0592910, upper bound: 70.0592932
time: 6.61 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0584102, upper bound: 70.0584071
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0584102, upper bound: 70.0584071
time: 6.31 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0345862, upper bound: 70.0345862
time: 10.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0345862, upper bound: 70.0345862
time: 8.48 seconds

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
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0400022, upper bound: 70.0400175
time: 8.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0400175, upper bound: 70.0400022
time: 7.13 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 17.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.20
Output dim: 8, lower bound: -70.0592913, upper bound: 70.0592928
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.20
Output dim: 8, lower bound: -70.0592910, upper bound: 70.0592932
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.20
Output dim: 8, lower bound: -70.0584102, upper bound: 70.0584071
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.20
Output dim: 8, lower bound: -70.0584102, upper bound: 70.0584071
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.20
Output dim: 8, lower bound: -70.0345862, upper bound: 70.0345862
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.20
Output dim: 8, lower bound: -70.0345862, upper bound: 70.0345862
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 17.20
Output dim: 8, lower bound: -70.0400022, upper bound: 70.0400175
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 17.20
Output dim: 8, lower bound: -70.0400175, upper bound: 70.0400022

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
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0592913, upper bound: 70.0592932
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0592913, upper bound: 70.0592932
time: 7.44 seconds

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
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0444404, upper bound: 70.0444420
time: 8.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0444404, upper bound: 70.0444420
time: 7.00 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0548342, upper bound: 70.0548342
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0548342, upper bound: 70.0548342
time: 7.39 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0095978, upper bound: 70.0095981
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0095978, upper bound: 70.0095981
time: 8.20 seconds

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
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0345835, upper bound: 70.0345835
time: 9.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0345835, upper bound: 70.0345835
time: 7.90 seconds

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
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0209958, upper bound: 70.0209958
time: 7.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0209958, upper bound: 70.0209958
time: 7.74 seconds

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0362493, upper bound: 70.0362515
time: 8.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0362493, upper bound: 70.0362515
time: 15.24 seconds

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
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0358870, upper bound: 70.0358904
time: 7.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0358927, upper bound: 70.0358861
time: 7.67 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 18.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.39
Output dim: 8, lower bound: -70.0592913, upper bound: 70.0592932
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.39
Output dim: 8, lower bound: -70.0592913, upper bound: 70.0592932
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.39
Output dim: 8, lower bound: -70.0444404, upper bound: 70.0444420
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.39
Output dim: 8, lower bound: -70.0444404, upper bound: 70.0444420
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.39
Output dim: 8, lower bound: -70.0548342, upper bound: 70.0548342
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.39
Output dim: 8, lower bound: -70.0548342, upper bound: 70.0548342
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 18.39
Output dim: 8, lower bound: -70.0095978, upper bound: 70.0095981
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 18.39
Output dim: 8, lower bound: -70.0095978, upper bound: 70.0095981
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.39
Output dim: 8, lower bound: -70.0345835, upper bound: 70.0345835
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.39
Output dim: 8, lower bound: -70.0345835, upper bound: 70.0345835
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 18.39
Output dim: 8, lower bound: -70.0209958, upper bound: 70.0209958
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 18.39
Output dim: 8, lower bound: -70.0209958, upper bound: 70.0209958
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.39
Output dim: 8, lower bound: -70.0362493, upper bound: 70.0362515
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.39
Output dim: 8, lower bound: -70.0362493, upper bound: 70.0362515
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.39
Output dim: 8, lower bound: -70.0358870, upper bound: 70.0358904
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.39
Output dim: 8, lower bound: -70.0358927, upper bound: 70.0358861

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0500967, upper bound: 70.0500994
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0500980, upper bound: 70.0500985
time: 7.69 seconds

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
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0585708, upper bound: 70.0585165
time: 10.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0585164, upper bound: 70.0585708
time: 7.75 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0438843, upper bound: 70.0438838
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0438835, upper bound: 70.0438845
time: 7.04 seconds

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0196765, upper bound: 70.0196775
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0196765, upper bound: 70.0196775
time: 6.11 seconds

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
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0547566, upper bound: 70.0547561
time: 15.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0547566, upper bound: 70.0547561
time: 15.37 seconds

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
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0548330, upper bound: 70.0548342
time: 8.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0548342, upper bound: 70.0548325
time: 9.23 seconds

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
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0345835, upper bound: 70.0345805
time: 7.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0345805, upper bound: 70.0345835
time: 8.55 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0209958, upper bound: 70.0209958
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0209958, upper bound: 70.0209958
time: 7.17 seconds

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
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 183

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0354841, upper bound: 70.0354689
time: 8.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0354636, upper bound: 70.0354855
time: 7.63 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0362485, upper bound: 70.0362501
time: 8.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0362485, upper bound: 70.0362501
time: 8.17 seconds

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
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0199837, upper bound: 70.0199888
time: 8.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0199837, upper bound: 70.0199888
time: 8.25 seconds

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0357495, upper bound: 70.0357395
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0357495, upper bound: 70.0357395
time: 8.15 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0500967, upper bound: 70.0500994
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0500980, upper bound: 70.0500985
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0585708, upper bound: 70.0585165
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0585164, upper bound: 70.0585708
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0438843, upper bound: 70.0438838
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0438835, upper bound: 70.0438845
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0196765, upper bound: 70.0196775
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0196765, upper bound: 70.0196775
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0547566, upper bound: 70.0547561
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0547566, upper bound: 70.0547561
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0548330, upper bound: 70.0548342
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0548342, upper bound: 70.0548325
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0345835, upper bound: 70.0345805
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0345805, upper bound: 70.0345835
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0209958, upper bound: 70.0209958
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0209958, upper bound: 70.0209958
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0354841, upper bound: 70.0354689
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0354636, upper bound: 70.0354855
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0362485, upper bound: 70.0362501
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0362485, upper bound: 70.0362501
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0199837, upper bound: 70.0199888
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0199837, upper bound: 70.0199888
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0357495, upper bound: 70.0357395
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.77
Output dim: 8, lower bound: -70.0357495, upper bound: 70.0357395

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0463022, upper bound: 70.0463099
time: 8.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0463105, upper bound: 70.0463018
time: 9.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0500926, upper bound: 70.0500936
time: 8.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0500926, upper bound: 70.0500936
time: 9.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0577915, upper bound: 70.0577508
time: 8.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0577915, upper bound: 70.0577508
time: 7.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0545443, upper bound: 70.0545796
time: 7.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0545534, upper bound: 70.0545691
time: 7.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=82.72178649902344
rel_dist={8: [-70.0816870801838, 70.08168706258579]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0365039, upper bound: 70.0365039
time: 8.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0365039, upper bound: 70.0365039
time: 8.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.41 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.41
Output dim: 8, lower bound: -70.0365039, upper bound: 70.0365039
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.41
Output dim: 8, lower bound: -70.0365039, upper bound: 70.0365039

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0365039, upper bound: 70.0364959
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0364959, upper bound: 70.0365039
time: 9.95 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0357574, upper bound: 70.0357574
time: 9.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0357574, upper bound: 70.0357574
time: 6.83 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.42 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.42
Output dim: 8, lower bound: -70.0365039, upper bound: 70.0364959
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.42
Output dim: 8, lower bound: -70.0364959, upper bound: 70.0365039
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.42
Output dim: 8, lower bound: -70.0357574, upper bound: 70.0357574
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.42
Output dim: 8, lower bound: -70.0357574, upper bound: 70.0357574

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0365017, upper bound: 70.0364959
time: 8.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0364959, upper bound: 70.0364946
time: 7.36 seconds

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
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0364959, upper bound: 70.0365038
time: 6.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0364959, upper bound: 70.0365039
time: 8.03 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0288026, upper bound: 70.0288019
time: 12.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0288019, upper bound: 70.0288026
time: 8.08 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0337201, upper bound: 70.0337201
time: 9.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0337201, upper bound: 70.0337201
time: 17.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 28.88 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.88
Output dim: 8, lower bound: -70.0365017, upper bound: 70.0364959
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.88
Output dim: 8, lower bound: -70.0364959, upper bound: 70.0364946
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.88
Output dim: 8, lower bound: -70.0364959, upper bound: 70.0365038
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.88
Output dim: 8, lower bound: -70.0364959, upper bound: 70.0365039
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.88
Output dim: 8, lower bound: -70.0288026, upper bound: 70.0288019
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.88
Output dim: 8, lower bound: -70.0288019, upper bound: 70.0288026
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.88
Output dim: 8, lower bound: -70.0337201, upper bound: 70.0337201
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.88
Output dim: 8, lower bound: -70.0337201, upper bound: 70.0337201

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
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0237432, upper bound: 70.0237398
time: 8.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0237432, upper bound: 70.0237392
time: 8.21 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0251477, upper bound: 70.0251560
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0251477, upper bound: 70.0251560
time: 7.51 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0251524, upper bound: 70.0251555
time: 7.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0251524, upper bound: 70.0251555
time: 9.13 seconds

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
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0315579, upper bound: 70.0315587
time: 8.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0315582, upper bound: 70.0315586
time: 8.03 seconds

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
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0287175, upper bound: 70.0287174
time: 8.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0287175, upper bound: 70.0287174
time: 7.72 seconds

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
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0240004, upper bound: 70.0240010
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0240004, upper bound: 70.0240010
time: 7.03 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0286379, upper bound: 70.0286447
time: 8.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0286447, upper bound: 70.0286379
time: 9.03 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0286379, upper bound: 70.0286447
time: 7.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0286447, upper bound: 70.0286379
time: 9.64 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 18.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 18.01
Output dim: 8, lower bound: -70.0237432, upper bound: 70.0237398
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 18.01
Output dim: 8, lower bound: -70.0237432, upper bound: 70.0237392
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.01
Output dim: 8, lower bound: -70.0251477, upper bound: 70.0251560
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.01
Output dim: 8, lower bound: -70.0251477, upper bound: 70.0251560
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.01
Output dim: 8, lower bound: -70.0251524, upper bound: 70.0251555
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.01
Output dim: 8, lower bound: -70.0251524, upper bound: 70.0251555
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.01
Output dim: 8, lower bound: -70.0315579, upper bound: 70.0315587
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.01
Output dim: 8, lower bound: -70.0315582, upper bound: 70.0315586
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.01
Output dim: 8, lower bound: -70.0287175, upper bound: 70.0287174
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.01
Output dim: 8, lower bound: -70.0287175, upper bound: 70.0287174
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 18.01
Output dim: 8, lower bound: -70.0240004, upper bound: 70.0240010
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 18.01
Output dim: 8, lower bound: -70.0240004, upper bound: 70.0240010
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.01
Output dim: 8, lower bound: -70.0286379, upper bound: 70.0286447
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.01
Output dim: 8, lower bound: -70.0286447, upper bound: 70.0286379
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.01
Output dim: 8, lower bound: -70.0286379, upper bound: 70.0286447
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.01
Output dim: 8, lower bound: -70.0286447, upper bound: 70.0286379

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0251426, upper bound: 70.0251560
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0251477, upper bound: 70.0251524
time: 7.26 seconds

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
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -69.9963134, upper bound: 69.9963180
time: 6.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -69.9963134, upper bound: 69.9963180
time: 5.73 seconds

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0173806, upper bound: 70.0173841
time: 27.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0173806, upper bound: 70.0173841
time: 20.11 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0187089, upper bound: 70.0187119
time: 7.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0187097, upper bound: 70.0187110
time: 8.23 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0080882, upper bound: 70.0080838
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0080882, upper bound: 70.0080838
time: 6.48 seconds

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
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0308226, upper bound: 70.0308245
time: 7.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0308226, upper bound: 70.0308245
time: 8.47 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0166750, upper bound: 70.0166768
time: 6.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0166750, upper bound: 70.0166768
time: 8.89 seconds

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0287175, upper bound: 70.0287173
time: 8.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0287173, upper bound: 70.0287174
time: 8.08 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0286379, upper bound: 70.0286423
time: 7.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0286348, upper bound: 70.0286447
time: 8.15 seconds

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
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0216764, upper bound: 70.0216763
time: 7.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0216763, upper bound: 70.0216764
time: 8.24 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0219973, upper bound: 70.0220027
time: 8.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0219966, upper bound: 70.0220027
time: 9.60 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0286447, upper bound: 70.0286374
time: 8.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0286435, upper bound: 70.0286379
time: 8.74 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 18.23 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0251426, upper bound: 70.0251560
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0251477, upper bound: 70.0251524
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.23
Output dim: 8, lower bound: -69.9963134, upper bound: 69.9963180
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.23
Output dim: 8, lower bound: -69.9963134, upper bound: 69.9963180
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0173806, upper bound: 70.0173841
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0173806, upper bound: 70.0173841
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0187089, upper bound: 70.0187119
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0187097, upper bound: 70.0187110
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0080882, upper bound: 70.0080838
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0080882, upper bound: 70.0080838
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0308226, upper bound: 70.0308245
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0308226, upper bound: 70.0308245
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0166750, upper bound: 70.0166768
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0166750, upper bound: 70.0166768
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0287175, upper bound: 70.0287173
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0287173, upper bound: 70.0287174
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0286379, upper bound: 70.0286423
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0286348, upper bound: 70.0286447
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0216764, upper bound: 70.0216763
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0216763, upper bound: 70.0216764
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0219973, upper bound: 70.0220027
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0219966, upper bound: 70.0220027
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0286447, upper bound: 70.0286374
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.23
Output dim: 8, lower bound: -70.0286435, upper bound: 70.0286379

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0251417, upper bound: 70.0251560
time: 29.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0251426, upper bound: 70.0251560
time: 12.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -69.9979708, upper bound: 69.9979655
time: 7.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -69.9979708, upper bound: 69.9979655
time: 8.67 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 19.23 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.23
Output dim: 8, lower bound: -70.0251417, upper bound: 70.0251560
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.23
Output dim: 8, lower bound: -70.0251426, upper bound: 70.0251560
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 19.23
Output dim: 8, lower bound: -69.9979708, upper bound: 69.9979655
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 19.23
Output dim: 8, lower bound: -69.9979708, upper bound: 69.9979655
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.23
Output dim: 8, lower bound: -70.0308226, upper bound: 70.0308245
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.23
Output dim: 8, lower bound: -70.0308226, upper bound: 70.0308245
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.23
Output dim: 8, lower bound: -70.0287175, upper bound: 70.0287173
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.23
Output dim: 8, lower bound: -70.0287173, upper bound: 70.0287174
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.23
Output dim: 8, lower bound: -70.0286379, upper bound: 70.0286423
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.23
Output dim: 8, lower bound: -70.0286348, upper bound: 70.0286447
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.23
Output dim: 8, lower bound: -70.0286447, upper bound: 70.0286374
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.23
Output dim: 8, lower bound: -70.0286435, upper bound: 70.0286379
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=82.72178649902344
rel_dist={8: [-70.08157121987698, 70.08157121987699]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0186818, upper bound: 70.0186818
time: 7.54 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0186818, upper bound: 70.0186818
time: 7.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.10 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 15.10
Output dim: 8, lower bound: -70.0186818, upper bound: 70.0186818
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 15.10
Output dim: 8, lower bound: -70.0186818, upper bound: 70.0186818
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=82.72178649902344
rel_dist={8: [-70.08140925617062, 70.08140925618659]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0815107, upper bound: 70.0815118
time: 9.76 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0815118, upper bound: 70.0815107
time: 8.94 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 18.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 18.72
Output dim: 8, lower bound: -70.0815107, upper bound: 70.0815118
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 18.72
Output dim: 8, lower bound: -70.0815118, upper bound: 70.0815107

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
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0651033, upper bound: 70.0651038
time: 10.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0651033, upper bound: 70.0651038
time: 8.59 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0736636, upper bound: 70.0736566
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0736565, upper bound: 70.0736641
time: 8.03 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.21 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.21
Output dim: 8, lower bound: -70.0651033, upper bound: 70.0651038
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.21
Output dim: 8, lower bound: -70.0651033, upper bound: 70.0651038
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.21
Output dim: 8, lower bound: -70.0736636, upper bound: 70.0736566
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.21
Output dim: 8, lower bound: -70.0736565, upper bound: 70.0736641

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
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0651031, upper bound: 70.0651038
time: 10.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0651033, upper bound: 70.0651036
time: 9.51 seconds

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0616477, upper bound: 70.0616623
time: 7.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0616616, upper bound: 70.0616466
time: 8.67 seconds

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
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0067781, upper bound: 70.0067789
time: 7.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0067781, upper bound: 70.0067789
time: 7.93 seconds

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
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0237379, upper bound: 70.0237384
time: 7.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0237379, upper bound: 70.0237384
time: 8.93 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.02 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.02
Output dim: 8, lower bound: -70.0651031, upper bound: 70.0651038
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.02
Output dim: 8, lower bound: -70.0651033, upper bound: 70.0651036
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.02
Output dim: 8, lower bound: -70.0616477, upper bound: 70.0616623
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.02
Output dim: 8, lower bound: -70.0616616, upper bound: 70.0616466
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.02
Output dim: 8, lower bound: -70.0067781, upper bound: 70.0067789
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.02
Output dim: 8, lower bound: -70.0067781, upper bound: 70.0067789
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.02
Output dim: 8, lower bound: -70.0237379, upper bound: 70.0237384
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.02
Output dim: 8, lower bound: -70.0237379, upper bound: 70.0237384

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0616476, upper bound: 70.0616617
time: 9.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0616616, upper bound: 70.0616464
time: 7.87 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0647640, upper bound: 70.0647653
time: 8.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0647640, upper bound: 70.0647653
time: 8.34 seconds

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
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0616477, upper bound: 70.0616623
time: 8.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0616477, upper bound: 70.0616623
time: 7.42 seconds

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0616616, upper bound: 70.0616464
time: 9.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0616616, upper bound: 70.0616466
time: 9.03 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 19.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.64
Output dim: 8, lower bound: -70.0616476, upper bound: 70.0616617
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.64
Output dim: 8, lower bound: -70.0616616, upper bound: 70.0616464
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.64
Output dim: 8, lower bound: -70.0647640, upper bound: 70.0647653
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.64
Output dim: 8, lower bound: -70.0647640, upper bound: 70.0647653
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.64
Output dim: 8, lower bound: -70.0616477, upper bound: 70.0616623
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.64
Output dim: 8, lower bound: -70.0616477, upper bound: 70.0616623
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 19.64
Output dim: 8, lower bound: -70.0616616, upper bound: 70.0616464
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 19.64
Output dim: 8, lower bound: -70.0616616, upper bound: 70.0616466

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0335613, upper bound: 70.0335641
time: 7.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0335613, upper bound: 70.0335641
time: 7.63 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0540034, upper bound: 70.0539958
time: 6.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0540028, upper bound: 70.0539969
time: 11.41 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0375720, upper bound: 70.0375699
time: 8.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0375720, upper bound: 70.0375699
time: 7.73 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0640620, upper bound: 70.0640619
time: 9.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0640620, upper bound: 70.0640619
time: 10.88 seconds

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
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0539967, upper bound: 70.0540056
time: 8.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0539958, upper bound: 70.0540056
time: 8.29 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0612216, upper bound: 70.0612370
time: 9.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0612199, upper bound: 70.0612380
time: 8.72 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0612784, upper bound: 70.0612603
time: 8.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0612784, upper bound: 70.0612603
time: 9.35 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0585123, upper bound: 70.0585050
time: 8.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0585123, upper bound: 70.0585050
time: 8.66 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.38
Output dim: 8, lower bound: -70.0335613, upper bound: 70.0335641
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.38
Output dim: 8, lower bound: -70.0335613, upper bound: 70.0335641
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.38
Output dim: 8, lower bound: -70.0540034, upper bound: 70.0539958
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.38
Output dim: 8, lower bound: -70.0540028, upper bound: 70.0539969
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.38
Output dim: 8, lower bound: -70.0375720, upper bound: 70.0375699
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.38
Output dim: 8, lower bound: -70.0375720, upper bound: 70.0375699
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.38
Output dim: 8, lower bound: -70.0640620, upper bound: 70.0640619
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.38
Output dim: 8, lower bound: -70.0640620, upper bound: 70.0640619
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.38
Output dim: 8, lower bound: -70.0539967, upper bound: 70.0540056
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.38
Output dim: 8, lower bound: -70.0539958, upper bound: 70.0540056
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.38
Output dim: 8, lower bound: -70.0612216, upper bound: 70.0612370
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.38
Output dim: 8, lower bound: -70.0612199, upper bound: 70.0612380
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.38
Output dim: 8, lower bound: -70.0612784, upper bound: 70.0612603
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.38
Output dim: 8, lower bound: -70.0612784, upper bound: 70.0612603
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.38
Output dim: 8, lower bound: -70.0585123, upper bound: 70.0585050
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.38
Output dim: 8, lower bound: -70.0585123, upper bound: 70.0585050

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0335613, upper bound: 70.0335641
time: 7.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0335613, upper bound: 70.0335639
time: 6.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0335613, upper bound: 70.0335641
time: 7.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0335613, upper bound: 70.0335639
time: 6.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0439277, upper bound: 70.0439274
time: 7.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0439277, upper bound: 70.0439274
time: 7.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0256888, upper bound: 70.0256944
time: 8.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0256888, upper bound: 70.0256944
time: 7.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0375716, upper bound: 70.0375699
time: 8.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0375720, upper bound: 70.0375698
time: 9.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0323869, upper bound: 70.0323859
time: 8.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0323865, upper bound: 70.0323865
time: 11.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0634735, upper bound: 70.0634714
time: 9.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0634711, upper bound: 70.0634735
time: 8.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0364100, upper bound: 70.0364140
time: 9.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0364100, upper bound: 70.0364140
time: 9.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0539963, upper bound: 70.0540056
time: 10.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0539967, upper bound: 70.0540051
time: 9.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0506060, upper bound: 70.0506150
time: 15.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0506060, upper bound: 70.0506150
time: 9.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 183
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0494283, upper bound: 70.0494252
time: 8.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0494228, upper bound: 70.0494255
time: 9.51 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 18.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0335613, upper bound: 70.0335641
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0335613, upper bound: 70.0335639
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0335613, upper bound: 70.0335641
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0335613, upper bound: 70.0335639
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0439277, upper bound: 70.0439274
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0439277, upper bound: 70.0439274
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0256888, upper bound: 70.0256944
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0256888, upper bound: 70.0256944
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0375716, upper bound: 70.0375699
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0375720, upper bound: 70.0375698
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0323869, upper bound: 70.0323859
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0323865, upper bound: 70.0323865
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0634735, upper bound: 70.0634714
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0634711, upper bound: 70.0634735
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0364100, upper bound: 70.0364140
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0364100, upper bound: 70.0364140
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0539963, upper bound: 70.0540056
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0539967, upper bound: 70.0540051
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0506060, upper bound: 70.0506150
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0506060, upper bound: 70.0506150
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0494283, upper bound: 70.0494252
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 18.74
Output dim: 8, lower bound: -70.0494228, upper bound: 70.0494255
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.74
Output dim: 8, lower bound: -70.0612199, upper bound: 70.0612380
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.74
Output dim: 8, lower bound: -70.0612784, upper bound: 70.0612603
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.74
Output dim: 8, lower bound: -70.0612784, upper bound: 70.0612603
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.74
Output dim: 8, lower bound: -70.0585123, upper bound: 70.0585050
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.74
Output dim: 8, lower bound: -70.0585123, upper bound: 70.0585050
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=82.72178649902344
rel_dist={8: [-70.08151177770955, 70.08151177770955]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 1856.67 seconds
