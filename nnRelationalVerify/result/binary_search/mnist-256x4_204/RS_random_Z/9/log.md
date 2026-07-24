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
execution time: IAR + LP analysis = 1.42 + 8.52 = 9.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -33.8800858, upper bound: 33.8800858


# Binary Search by BASE starts (time budget: 2690.06 seconds, max iter: 100)

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
Binary search time: 24.42 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2665.64 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8790125, upper bound: 33.8790269
time: 3.97 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8790269, upper bound: 33.8790124
time: 3.45 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.44 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.44
Output dim: 9, lower bound: -33.8790125, upper bound: 33.8790269
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.44
Output dim: 9, lower bound: -33.8790269, upper bound: 33.8790124

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8784575, upper bound: 33.8784769
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8784669, upper bound: 33.8784679
time: 12.46 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 168

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8782186, upper bound: 33.8782061
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8782187, upper bound: 33.8782065
time: 3.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 10.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 10.12
Output dim: 9, lower bound: -33.8784575, upper bound: 33.8784769
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 10.12
Output dim: 9, lower bound: -33.8784669, upper bound: 33.8784679
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 10.12
Output dim: 9, lower bound: -33.8782186, upper bound: 33.8782061
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 10.12
Output dim: 9, lower bound: -33.8782187, upper bound: 33.8782065

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8769391, upper bound: 33.8769448
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8769391, upper bound: 33.8769448
time: 4.90 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8762379, upper bound: 33.8762326
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8762377, upper bound: 33.8762326
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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8780122, upper bound: 33.8780011
time: 2.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8780122, upper bound: 33.8780017
time: 19.18 seconds

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
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8671980, upper bound: 33.8671974
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8671980, upper bound: 33.8671974
time: 4.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 9.81 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.81
Output dim: 9, lower bound: -33.8769391, upper bound: 33.8769448
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.81
Output dim: 9, lower bound: -33.8769391, upper bound: 33.8769448
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.81
Output dim: 9, lower bound: -33.8762379, upper bound: 33.8762326
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.81
Output dim: 9, lower bound: -33.8762377, upper bound: 33.8762326
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.81
Output dim: 9, lower bound: -33.8780122, upper bound: 33.8780011
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.81
Output dim: 9, lower bound: -33.8780122, upper bound: 33.8780017
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 9.81
Output dim: 9, lower bound: -33.8671980, upper bound: 33.8671974
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 9.81
Output dim: 9, lower bound: -33.8671980, upper bound: 33.8671974

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8644467, upper bound: 33.8644490
time: 3.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8644467, upper bound: 33.8644490
time: 3.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8736355, upper bound: 33.8736364
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8736355, upper bound: 33.8736365
time: 5.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8762379, upper bound: 33.8762324
time: 9.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8762371, upper bound: 33.8762326
time: 4.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8745190, upper bound: 33.8745070
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8745175, upper bound: 33.8745090
time: 4.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8726656, upper bound: 33.8726589
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8726656, upper bound: 33.8726589
time: 3.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8508076, upper bound: 33.8508089
time: 40.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8508076, upper bound: 33.8508091
time: 5.93 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 47.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 47.88
Output dim: 9, lower bound: -33.8644467, upper bound: 33.8644490
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 47.88
Output dim: 9, lower bound: -33.8644467, upper bound: 33.8644490
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 47.88
Output dim: 9, lower bound: -33.8736355, upper bound: 33.8736364
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 47.88
Output dim: 9, lower bound: -33.8736355, upper bound: 33.8736365
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 47.88
Output dim: 9, lower bound: -33.8762379, upper bound: 33.8762324
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 47.88
Output dim: 9, lower bound: -33.8762371, upper bound: 33.8762326
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 47.88
Output dim: 9, lower bound: -33.8745190, upper bound: 33.8745070
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 47.88
Output dim: 9, lower bound: -33.8745175, upper bound: 33.8745090
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 47.88
Output dim: 9, lower bound: -33.8726656, upper bound: 33.8726589
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 47.88
Output dim: 9, lower bound: -33.8726656, upper bound: 33.8726589
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 47.88
Output dim: 9, lower bound: -33.8508076, upper bound: 33.8508089
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 47.88
Output dim: 9, lower bound: -33.8508076, upper bound: 33.8508091

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8741134, upper bound: 33.8741111
time: 4.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8741134, upper bound: 33.8741110
time: 4.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8716369, upper bound: 33.8716434
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8716369, upper bound: 33.8716434
time: 3.45 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 9.34 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 9.34
Output dim: 9, lower bound: -33.8741134, upper bound: 33.8741111
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 9.34
Output dim: 9, lower bound: -33.8741134, upper bound: 33.8741110
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 9.34
Output dim: 9, lower bound: -33.8716369, upper bound: 33.8716434
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 9.34
Output dim: 9, lower bound: -33.8716369, upper bound: 33.8716434
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=41.571006774902344
rel_dist={9: [-33.88006301494167, 33.88006301377598]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8789158, upper bound: 33.8789141
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8789141, upper bound: 33.8789158
time: 4.08 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.63 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.63
Output dim: 9, lower bound: -33.8789158, upper bound: 33.8789141
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.63
Output dim: 9, lower bound: -33.8789141, upper bound: 33.8789158

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
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8783869, upper bound: 33.8783911
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8783915, upper bound: 33.8783854
time: 2.96 seconds

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
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8774695, upper bound: 33.8774709
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8774695, upper bound: 33.8774715
time: 4.58 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 10.82 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 10.82
Output dim: 9, lower bound: -33.8783869, upper bound: 33.8783911
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 10.82
Output dim: 9, lower bound: -33.8783915, upper bound: 33.8783854
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 10.82
Output dim: 9, lower bound: -33.8774695, upper bound: 33.8774709
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 10.82
Output dim: 9, lower bound: -33.8774695, upper bound: 33.8774715

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8546706, upper bound: 33.8546684
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8546706, upper bound: 33.8546684
time: 5.60 seconds

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
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 129

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8669532, upper bound: 33.8669490
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8669532, upper bound: 33.8669490
time: 3.76 seconds

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
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8710301, upper bound: 33.8710310
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8710301, upper bound: 33.8710309
time: 4.07 seconds

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
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8748005, upper bound: 33.8748040
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8748003, upper bound: 33.8748038
time: 4.15 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 9.93 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 9.93
Output dim: 9, lower bound: -33.8546706, upper bound: 33.8546684
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 9.93
Output dim: 9, lower bound: -33.8546706, upper bound: 33.8546684
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 9.93
Output dim: 9, lower bound: -33.8669532, upper bound: 33.8669490
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 9.93
Output dim: 9, lower bound: -33.8669532, upper bound: 33.8669490
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 9.93
Output dim: 9, lower bound: -33.8710301, upper bound: 33.8710310
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 9.93
Output dim: 9, lower bound: -33.8710301, upper bound: 33.8710309
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.93
Output dim: 9, lower bound: -33.8748005, upper bound: 33.8748040
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.93
Output dim: 9, lower bound: -33.8748003, upper bound: 33.8748038

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8719230, upper bound: 33.8719199
time: 3.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8719203, upper bound: 33.8719333
time: 3.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8748002, upper bound: 33.8747964
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8747935, upper bound: 33.8748038
time: 3.59 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 11.39 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 11.39
Output dim: 9, lower bound: -33.8719230, upper bound: 33.8719199
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 11.39
Output dim: 9, lower bound: -33.8719203, upper bound: 33.8719333
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.39
Output dim: 9, lower bound: -33.8748002, upper bound: 33.8747964
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.39
Output dim: 9, lower bound: -33.8747935, upper bound: 33.8748038

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8728878, upper bound: 33.8728873
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8728878, upper bound: 33.8728873
time: 2.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 168

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8485177, upper bound: 33.8485192
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8485177, upper bound: 33.8485192
time: 4.16 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 9.47 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 9.47
Output dim: 9, lower bound: -33.8728878, upper bound: 33.8728873
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 9.47
Output dim: 9, lower bound: -33.8728878, upper bound: 33.8728873
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 9.47
Output dim: 9, lower bound: -33.8485177, upper bound: 33.8485192
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 9.47
Output dim: 9, lower bound: -33.8485177, upper bound: 33.8485192
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=41.571006774902344
rel_dist={9: [-33.88007495180708, 33.88007494630283]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8735771, upper bound: 33.8735771
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8735771, upper bound: 33.8735771
time: 3.42 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.24 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 7.24
Output dim: 9, lower bound: -33.8735771, upper bound: 33.8735771
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 7.24
Output dim: 9, lower bound: -33.8735771, upper bound: 33.8735771
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=41.571006774902344
rel_dist={9: [-33.88008219383614, 33.88008219146937]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 129
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8503070, upper bound: 33.8503070
time: 2.39 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8503070, upper bound: 33.8503070
time: 2.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 4.79 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 4.79
Output dim: 9, lower bound: -33.8503070, upper bound: 33.8503070
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 4.79
Output dim: 9, lower bound: -33.8503070, upper bound: 33.8503070
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=41.571006774902344
rel_dist={9: [-33.88008576028902, 33.88008575296402]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 366.29 seconds
