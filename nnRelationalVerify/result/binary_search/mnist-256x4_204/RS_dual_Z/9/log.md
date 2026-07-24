## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 33.8747337875
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899)
1: (-17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774)
2: (-22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451)
3: (-28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570)
4: (-25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998)
5: (-21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947)
6: (-20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817)
7: (-24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103)
8: (-28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781)
9: (-23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068)

## BASE Result
execution time: IAR + LP analysis = 1.23 + 8.37 = 9.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -33.8800858, upper bound: 33.8800858


# Binary Search by BASE starts (time budget: 2690.40 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=41.571006774902344
rel_dist={9: [-33.88006301494167, 33.88006301377598]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=41.571006774902344
rel_dist={9: [-33.88001792224934, 33.88001792519913]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=41.571006774902344
rel_dist={9: [-33.879905218442005, 33.87990521540584]}

## Binary Search Result
Binary search time: 24.07 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2666.33 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8795378, upper bound: 33.8795380
time: 4.15 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8795380, upper bound: 33.8795378
time: 5.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.80 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.80
Output dim: 9, lower bound: -33.8795378, upper bound: 33.8795380
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.80
Output dim: 9, lower bound: -33.8795380, upper bound: 33.8795378

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8774719, upper bound: 33.8774722
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8774717, upper bound: 33.8774728
time: 4.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8774727, upper bound: 33.8774716
time: 3.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8774726, upper bound: 33.8774716
time: 5.65 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 10.34 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 10.34
Output dim: 9, lower bound: -33.8774719, upper bound: 33.8774722
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 10.34
Output dim: 9, lower bound: -33.8774717, upper bound: 33.8774728
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 10.34
Output dim: 9, lower bound: -33.8774727, upper bound: 33.8774716
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 10.34
Output dim: 9, lower bound: -33.8774726, upper bound: 33.8774716

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695025, upper bound: 33.8695037
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695025, upper bound: 33.8695037
time: 7.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695034, upper bound: 33.8695027
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695034, upper bound: 33.8695027
time: 4.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695028, upper bound: 33.8695034
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695028, upper bound: 33.8695034
time: 17.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695037, upper bound: 33.8695025
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695037, upper bound: 33.8695025
time: 4.11 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 9.07 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 9.07
Output dim: 9, lower bound: -33.8695025, upper bound: 33.8695037
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 9.07
Output dim: 9, lower bound: -33.8695025, upper bound: 33.8695037
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 9.07
Output dim: 9, lower bound: -33.8695034, upper bound: 33.8695027
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 9.07
Output dim: 9, lower bound: -33.8695034, upper bound: 33.8695027
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 9.07
Output dim: 9, lower bound: -33.8695028, upper bound: 33.8695034
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 9.07
Output dim: 9, lower bound: -33.8695028, upper bound: 33.8695034
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 9.07
Output dim: 9, lower bound: -33.8695037, upper bound: 33.8695025
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 9.07
Output dim: 9, lower bound: -33.8695037, upper bound: 33.8695025
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=41.571006774902344
rel_dist={9: [-33.88006301494167, 33.88006301377598]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8795483, upper bound: 33.8795486
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8795486, upper bound: 33.8795483
time: 15.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 20.41 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 20.41
Output dim: 9, lower bound: -33.8795483, upper bound: 33.8795486
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 20.41
Output dim: 9, lower bound: -33.8795486, upper bound: 33.8795483

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8774937, upper bound: 33.8774943
time: 9.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8774933, upper bound: 33.8774945
time: 4.77 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8774950, upper bound: 33.8774933
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8774947, upper bound: 33.8774937
time: 4.72 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 10.70 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 10.70
Output dim: 9, lower bound: -33.8774937, upper bound: 33.8774943
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 10.70
Output dim: 9, lower bound: -33.8774933, upper bound: 33.8774945
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 10.70
Output dim: 9, lower bound: -33.8774950, upper bound: 33.8774933
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 10.70
Output dim: 9, lower bound: -33.8774947, upper bound: 33.8774937

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695125, upper bound: 33.8695140
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695125, upper bound: 33.8695141
time: 3.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695137, upper bound: 33.8695128
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695137, upper bound: 33.8695128
time: 3.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695128, upper bound: 33.8695137
time: 3.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695128, upper bound: 33.8695137
time: 3.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695140, upper bound: 33.8695125
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695140, upper bound: 33.8695125
time: 4.05 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 9.57 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 9.57
Output dim: 9, lower bound: -33.8695125, upper bound: 33.8695140
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 9.57
Output dim: 9, lower bound: -33.8695125, upper bound: 33.8695141
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 9.57
Output dim: 9, lower bound: -33.8695137, upper bound: 33.8695128
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 9.57
Output dim: 9, lower bound: -33.8695137, upper bound: 33.8695128
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 9.57
Output dim: 9, lower bound: -33.8695128, upper bound: 33.8695137
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 9.57
Output dim: 9, lower bound: -33.8695128, upper bound: 33.8695137
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 9.57
Output dim: 9, lower bound: -33.8695140, upper bound: 33.8695125
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 9.57
Output dim: 9, lower bound: -33.8695140, upper bound: 33.8695125
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=41.571006774902344
rel_dist={9: [-33.88007495180708, 33.88007494630283]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8795552, upper bound: 33.8795555
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8795555, upper bound: 33.8795552
time: 2.80 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.81
Output dim: 9, lower bound: -33.8795552, upper bound: 33.8795555
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.81
Output dim: 9, lower bound: -33.8795555, upper bound: 33.8795552

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8775079, upper bound: 33.8775090
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8775075, upper bound: 33.8775088
time: 2.77 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8775092, upper bound: 33.8775075
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8775091, upper bound: 33.8775080
time: 3.08 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 8.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 8.12
Output dim: 9, lower bound: -33.8775079, upper bound: 33.8775090
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 8.12
Output dim: 9, lower bound: -33.8775075, upper bound: 33.8775088
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 8.12
Output dim: 9, lower bound: -33.8775092, upper bound: 33.8775075
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 8.12
Output dim: 9, lower bound: -33.8775091, upper bound: 33.8775080

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695191, upper bound: 33.8695209
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695191, upper bound: 33.8695208
time: 7.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695206, upper bound: 33.8695196
time: 14.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695206, upper bound: 33.8695195
time: 3.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695195, upper bound: 33.8695206
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695196, upper bound: 33.8695206
time: 4.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695208, upper bound: 33.8695191
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695209, upper bound: 33.8695191
time: 3.91 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 9.96 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 9.96
Output dim: 9, lower bound: -33.8695191, upper bound: 33.8695209
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 9.96
Output dim: 9, lower bound: -33.8695191, upper bound: 33.8695208
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 9.96
Output dim: 9, lower bound: -33.8695206, upper bound: 33.8695196
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 9.96
Output dim: 9, lower bound: -33.8695206, upper bound: 33.8695195
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 9.96
Output dim: 9, lower bound: -33.8695195, upper bound: 33.8695206
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 9.96
Output dim: 9, lower bound: -33.8695196, upper bound: 33.8695206
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 9.96
Output dim: 9, lower bound: -33.8695208, upper bound: 33.8695191
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 9.96
Output dim: 9, lower bound: -33.8695209, upper bound: 33.8695191
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=41.571006774902344
rel_dist={9: [-33.88008219383614, 33.88008219146937]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8795587, upper bound: 33.8795591
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8795591, upper bound: 33.8795587
time: 3.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.42 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.42
Output dim: 9, lower bound: -33.8795587, upper bound: 33.8795591
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.42
Output dim: 9, lower bound: -33.8795591, upper bound: 33.8795587

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8775146, upper bound: 33.8775158
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8775142, upper bound: 33.8775159
time: 3.51 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8775159, upper bound: 33.8775141
time: 2.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8775158, upper bound: 33.8775141
time: 3.19 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 7.41 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.41
Output dim: 9, lower bound: -33.8775146, upper bound: 33.8775158
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.41
Output dim: 9, lower bound: -33.8775142, upper bound: 33.8775159
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.41
Output dim: 9, lower bound: -33.8775159, upper bound: 33.8775141
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.41
Output dim: 9, lower bound: -33.8775158, upper bound: 33.8775141

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695224, upper bound: 33.8695242
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695224, upper bound: 33.8695242
time: 2.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695240, upper bound: 33.8695229
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695240, upper bound: 33.8695229
time: 3.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695229, upper bound: 33.8695240
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695229, upper bound: 33.8695240
time: 4.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -23.2670193, 19.6152725, -23.2670193, 19.6152725, -42.8822899, 42.8822899
1: -17.2405739, 16.3867092, -17.2405739, 16.3867092, -33.6272812, 33.6272774
2: -22.8407764, 17.8757706, -22.8407764, 17.8757706, -40.7165451, 40.7165451
3: -28.6123161, 14.1993437, -28.6123161, 14.1993437, -42.8116570, 42.8116570
4: -25.6187172, 17.7315826, -25.6187172, 17.7315826, -43.3502998, 43.3502998
5: -21.6950798, 17.5882168, -21.6950798, 17.5882168, -39.2832947, 39.2832947
6: -20.7598629, 20.7582188, -20.7598629, 20.7582188, -41.5180817, 41.5180817
7: -24.2113094, 19.6866989, -24.2113094, 19.6866989, -43.8980103, 43.8980103
8: -28.6491680, 18.3628082, -28.6491680, 18.3628082, -47.0119781, 47.0119781
9: -23.9259472, 17.6450577, -23.9259472, 17.6450577, -41.5710030, 41.5710068

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695243, upper bound: 33.8695224
time: 2.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8695243, upper bound: 33.8695224
time: 3.20 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 7.44 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 7.44
Output dim: 9, lower bound: -33.8695224, upper bound: 33.8695242
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 7.44
Output dim: 9, lower bound: -33.8695224, upper bound: 33.8695242
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 7.44
Output dim: 9, lower bound: -33.8695240, upper bound: 33.8695229
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 7.44
Output dim: 9, lower bound: -33.8695240, upper bound: 33.8695229
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 7.44
Output dim: 9, lower bound: -33.8695229, upper bound: 33.8695240
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 7.44
Output dim: 9, lower bound: -33.8695229, upper bound: 33.8695240
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 7.44
Output dim: 9, lower bound: -33.8695243, upper bound: 33.8695224
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 7.44
Output dim: 9, lower bound: -33.8695243, upper bound: 33.8695224
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=41.571006774902344
rel_dist={9: [-33.88008576028902, 33.88008575296402]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 333.96 seconds
