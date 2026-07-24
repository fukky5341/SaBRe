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
execution time: IAR + LP analysis = 1.62 + 8.69 = 10.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -33.8800858, upper bound: 33.8800858


# Binary Search by BASE starts (time budget: 2689.69 seconds, max iter: 100)

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
Binary search time: 25.25 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2664.44 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8756630, upper bound: 33.8736912
time: 6.71 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8733187, upper bound: 33.8733187
time: 3.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.31 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.31
Output dim: 9, lower bound: -33.8756630, upper bound: 33.8736912
IS_A2, status: Status.VERIFIED, split count: 1, time: 10.31
Output dim: 9, lower bound: -33.8733187, upper bound: 33.8733187

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -18.4010162, 15.6114140, -23.2670193, 19.6152725, -38.0162849, 38.8784332
1: -13.5919094, 13.0152292, -17.2405739, 16.3867092, -29.9786186, 30.2557945
2: -17.9768524, 14.2358227, -22.8407764, 17.8757706, -35.8526230, 37.0765991
3: -22.5516376, 11.3189268, -28.6123161, 14.1993437, -36.7509804, 39.9312401
4: -20.2675476, 14.0601978, -25.6187172, 17.7315826, -37.9991302, 39.6789169
5: -17.1344261, 13.9622078, -21.6950798, 17.5882168, -34.7226410, 35.6572876
6: -16.3914318, 16.4827347, -20.7598629, 20.7582188, -37.1496468, 37.2425957
7: -19.1605721, 15.6680775, -24.2113094, 19.6866989, -38.8472710, 39.8793869
8: -22.6600533, 14.5711079, -28.6491680, 18.3628082, -41.0228615, 43.2202682
9: -19.0300560, 13.9656067, -23.9259472, 17.6450577, -36.6751137, 37.8915482

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 76

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8733187, upper bound: 33.8733187
time: 3.01 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8733187, upper bound: 33.8733187
time: 10.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.95 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 14.95
Output dim: 9, lower bound: -33.8733187, upper bound: 33.8733187
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 14.95
Output dim: 9, lower bound: -33.8733187, upper bound: 33.8733187
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=41.571006774902344
rel_dist={9: [-33.88006301494167, 33.88006301377598]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 76

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8761102, upper bound: 33.8738708
time: 3.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8733270, upper bound: 33.8733270
time: 3.41 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.05 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.05
Output dim: 9, lower bound: -33.8761102, upper bound: 33.8738708
IS_A2, status: Status.VERIFIED, split count: 1, time: 7.05
Output dim: 9, lower bound: -33.8733270, upper bound: 33.8733270

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -18.4010162, 15.6114140, -23.2670193, 19.6152725, -38.0162849, 38.8784332
1: -13.5919094, 13.0152292, -17.2405739, 16.3867092, -29.9786186, 30.2557945
2: -17.9768524, 14.2358227, -22.8407764, 17.8757706, -35.8526230, 37.0765991
3: -22.5516376, 11.3189268, -28.6123161, 14.1993437, -36.7509804, 39.9312401
4: -20.2675476, 14.0601978, -25.6187172, 17.7315826, -37.9991302, 39.6789169
5: -17.1344261, 13.9622078, -21.6950798, 17.5882168, -34.7226410, 35.6572876
6: -16.3914318, 16.4827347, -20.7598629, 20.7582188, -37.1496468, 37.2425957
7: -19.1605721, 15.6680775, -24.2113094, 19.6866989, -38.8472710, 39.8793869
8: -22.6600533, 14.5711079, -28.6491680, 18.3628082, -41.0228615, 43.2202682
9: -19.0300560, 13.9656067, -23.9259472, 17.6450577, -36.6751137, 37.8915482

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 76

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8733270, upper bound: 33.8733270
time: 3.07 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8733270, upper bound: 33.8733270
time: 3.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 8.16 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 8.16
Output dim: 9, lower bound: -33.8733270, upper bound: 33.8733270
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 8.16
Output dim: 9, lower bound: -33.8733270, upper bound: 33.8733270
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=41.571006774902344
rel_dist={9: [-33.88007495180708, 33.88007494630283]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 76

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8763217, upper bound: 33.8739835
time: 4.09 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8733325, upper bound: 33.8733325
time: 2.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 6.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 6.81
Output dim: 9, lower bound: -33.8763217, upper bound: 33.8739835
IS_A2, status: Status.VERIFIED, split count: 1, time: 6.81
Output dim: 9, lower bound: -33.8733325, upper bound: 33.8733325

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -18.4010162, 15.6114140, -23.2670193, 19.6152725, -38.0162849, 38.8784332
1: -13.5919094, 13.0152292, -17.2405739, 16.3867092, -29.9786186, 30.2557945
2: -17.9768524, 14.2358227, -22.8407764, 17.8757706, -35.8526230, 37.0765991
3: -22.5516376, 11.3189268, -28.6123161, 14.1993437, -36.7509804, 39.9312401
4: -20.2675476, 14.0601978, -25.6187172, 17.7315826, -37.9991302, 39.6789169
5: -17.1344261, 13.9622078, -21.6950798, 17.5882168, -34.7226410, 35.6572876
6: -16.3914318, 16.4827347, -20.7598629, 20.7582188, -37.1496468, 37.2425957
7: -19.1605721, 15.6680775, -24.2113094, 19.6866989, -38.8472710, 39.8793869
8: -22.6600533, 14.5711079, -28.6491680, 18.3628082, -41.0228615, 43.2202682
9: -19.0300560, 13.9656067, -23.9259472, 17.6450577, -36.6751137, 37.8915482

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 76

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8733325, upper bound: 33.8733325
time: 4.53 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8733325, upper bound: 33.8733325
time: 3.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 8.98 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 8.98
Output dim: 9, lower bound: -33.8733325, upper bound: 33.8733325
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 8.98
Output dim: 9, lower bound: -33.8733325, upper bound: 33.8733325
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=41.571006774902344
rel_dist={9: [-33.88008219383614, 33.88008219146937]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 76

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -33.8764010, upper bound: 33.8740372
time: 3.94 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8733352, upper bound: 33.8733352
time: 3.03 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.10 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.10
Output dim: 9, lower bound: -33.8764010, upper bound: 33.8740372
IS_A2, status: Status.VERIFIED, split count: 1, time: 7.10
Output dim: 9, lower bound: -33.8733352, upper bound: 33.8733352

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -18.4010162, 15.6114140, -23.2670193, 19.6152725, -38.0162849, 38.8784332
1: -13.5919094, 13.0152292, -17.2405739, 16.3867092, -29.9786186, 30.2557945
2: -17.9768524, 14.2358227, -22.8407764, 17.8757706, -35.8526230, 37.0765991
3: -22.5516376, 11.3189268, -28.6123161, 14.1993437, -36.7509804, 39.9312401
4: -20.2675476, 14.0601978, -25.6187172, 17.7315826, -37.9991302, 39.6789169
5: -17.1344261, 13.9622078, -21.6950798, 17.5882168, -34.7226410, 35.6572876
6: -16.3914318, 16.4827347, -20.7598629, 20.7582188, -37.1496468, 37.2425957
7: -19.1605721, 15.6680775, -24.2113094, 19.6866989, -38.8472710, 39.8793869
8: -22.6600533, 14.5711079, -28.6491680, 18.3628082, -41.0228615, 43.2202682
9: -19.0300560, 13.9656067, -23.9259472, 17.6450577, -36.6751137, 37.8915482

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 76

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8733352, upper bound: 33.8733352
time: 3.76 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -33.8733352, upper bound: 33.8733352
time: 3.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 9.03 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 9.03
Output dim: 9, lower bound: -33.8733352, upper bound: 33.8733352
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 9.03
Output dim: 9, lower bound: -33.8733352, upper bound: 33.8733352
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=41.571006774902344
rel_dist={9: [-33.88008576028902, 33.88008575296402]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 106.70 seconds
