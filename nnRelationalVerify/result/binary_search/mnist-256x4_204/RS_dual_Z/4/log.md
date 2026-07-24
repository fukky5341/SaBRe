## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 132.847234285
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868)
1: (-60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306)
2: (-78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985)
3: (-83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310)
4: (-78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780)
5: (-69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790)
6: (-64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515)
7: (-70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147)
8: (-90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884)
9: (-63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478)

## BASE Result
execution time: IAR + LP analysis = 1.16 + 10.52 = 11.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -132.9266407, upper bound: 132.9266406


# Binary Search by BASE starts (time budget: 2688.31 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=145.0349884033203
rel_dist={8: [-132.92649402057955, 132.9264940353584]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=145.0349884033203
rel_dist={8: [-132.92639238641755, 132.92639237568028]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=145.0349884033203
rel_dist={8: [-132.9262935075506, 132.92629350749155]}

## Binary Search Result
Binary search time: 34.99 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2653.33 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9203334, upper bound: 132.9202833
time: 6.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9202833, upper bound: 132.9203334
time: 8.49 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.17 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.17
Output dim: 8, lower bound: -132.9203334, upper bound: 132.9202833
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.17
Output dim: 8, lower bound: -132.9202833, upper bound: 132.9203334

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9023033, upper bound: 132.9022474
time: 8.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9022537, upper bound: 132.9022981
time: 6.10 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9022474, upper bound: 132.9022538
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9022474, upper bound: 132.9023033
time: 6.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.40 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.40
Output dim: 8, lower bound: -132.9023033, upper bound: 132.9022474
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.40
Output dim: 8, lower bound: -132.9022537, upper bound: 132.9022981
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.40
Output dim: 8, lower bound: -132.9022474, upper bound: 132.9022538
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.40
Output dim: 8, lower bound: -132.9022474, upper bound: 132.9023033

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734353, upper bound: 132.8733845
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734353, upper bound: 132.8733845
time: 6.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734003, upper bound: 132.8734226
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734003, upper bound: 132.8734226
time: 7.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734226, upper bound: 132.8734003
time: 6.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734226, upper bound: 132.8734003
time: 8.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8733845, upper bound: 132.8734353
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8733845, upper bound: 132.8734353
time: 6.63 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.31 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.31
Output dim: 8, lower bound: -132.8734353, upper bound: 132.8733845
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.31
Output dim: 8, lower bound: -132.8734353, upper bound: 132.8733845
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.31
Output dim: 8, lower bound: -132.8734003, upper bound: 132.8734226
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.31
Output dim: 8, lower bound: -132.8734003, upper bound: 132.8734226
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.31
Output dim: 8, lower bound: -132.8734226, upper bound: 132.8734003
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.31
Output dim: 8, lower bound: -132.8734226, upper bound: 132.8734003
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.31
Output dim: 8, lower bound: -132.8733845, upper bound: 132.8734353
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.31
Output dim: 8, lower bound: -132.8733845, upper bound: 132.8734353

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428756, upper bound: 132.8428697
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428756, upper bound: 132.8428697
time: 5.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428756, upper bound: 132.8428697
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428756, upper bound: 132.8428697
time: 9.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428703, upper bound: 132.8428752
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428703, upper bound: 132.8428752
time: 6.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428703, upper bound: 132.8428752
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428703, upper bound: 132.8428752
time: 6.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428752, upper bound: 132.8428703
time: 6.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428752, upper bound: 132.8428703
time: 5.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428752, upper bound: 132.8428703
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428752, upper bound: 132.8428703
time: 6.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428697, upper bound: 132.8428756
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428697, upper bound: 132.8428756
time: 6.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428697, upper bound: 132.8428756
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428697, upper bound: 132.8428756
time: 7.04 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.93
Output dim: 8, lower bound: -132.8428756, upper bound: 132.8428697
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.93
Output dim: 8, lower bound: -132.8428756, upper bound: 132.8428697
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.93
Output dim: 8, lower bound: -132.8428756, upper bound: 132.8428697
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.93
Output dim: 8, lower bound: -132.8428756, upper bound: 132.8428697
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.93
Output dim: 8, lower bound: -132.8428703, upper bound: 132.8428752
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.93
Output dim: 8, lower bound: -132.8428703, upper bound: 132.8428752
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.93
Output dim: 8, lower bound: -132.8428703, upper bound: 132.8428752
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.93
Output dim: 8, lower bound: -132.8428703, upper bound: 132.8428752
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.93
Output dim: 8, lower bound: -132.8428752, upper bound: 132.8428703
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.93
Output dim: 8, lower bound: -132.8428752, upper bound: 132.8428703
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.93
Output dim: 8, lower bound: -132.8428752, upper bound: 132.8428703
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.93
Output dim: 8, lower bound: -132.8428752, upper bound: 132.8428703
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.93
Output dim: 8, lower bound: -132.8428697, upper bound: 132.8428756
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.93
Output dim: 8, lower bound: -132.8428697, upper bound: 132.8428756
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.93
Output dim: 8, lower bound: -132.8428697, upper bound: 132.8428756
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.93
Output dim: 8, lower bound: -132.8428697, upper bound: 132.8428756
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=145.0349884033203
rel_dist={8: [-132.92649402057955, 132.9264940353584]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9204042, upper bound: 132.9203341
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9203341, upper bound: 132.9204042
time: 7.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.33 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.33
Output dim: 8, lower bound: -132.9204042, upper bound: 132.9203341
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.33
Output dim: 8, lower bound: -132.9203341, upper bound: 132.9204042

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9023832, upper bound: 132.9022955
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9023050, upper bound: 132.9023756
time: 5.94 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9023756, upper bound: 132.9023050
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9022955, upper bound: 132.9023832
time: 9.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.60 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.60
Output dim: 8, lower bound: -132.9023832, upper bound: 132.9022955
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.60
Output dim: 8, lower bound: -132.9023050, upper bound: 132.9023756
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.60
Output dim: 8, lower bound: -132.9023756, upper bound: 132.9023050
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.60
Output dim: 8, lower bound: -132.9022955, upper bound: 132.9023832

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734836, upper bound: 132.8734085
time: 7.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734836, upper bound: 132.8734085
time: 6.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734286, upper bound: 132.8734646
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734286, upper bound: 132.8734646
time: 4.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734646, upper bound: 132.8734286
time: 7.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734646, upper bound: 132.8734286
time: 7.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734085, upper bound: 132.8734836
time: 6.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734085, upper bound: 132.8734836
time: 6.94 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.83 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.83
Output dim: 8, lower bound: -132.8734836, upper bound: 132.8734085
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.83
Output dim: 8, lower bound: -132.8734836, upper bound: 132.8734085
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.83
Output dim: 8, lower bound: -132.8734286, upper bound: 132.8734646
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.83
Output dim: 8, lower bound: -132.8734286, upper bound: 132.8734646
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.83
Output dim: 8, lower bound: -132.8734646, upper bound: 132.8734286
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.83
Output dim: 8, lower bound: -132.8734646, upper bound: 132.8734286
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.83
Output dim: 8, lower bound: -132.8734085, upper bound: 132.8734836
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.83
Output dim: 8, lower bound: -132.8734085, upper bound: 132.8734836

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428982, upper bound: 132.8428891
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428982, upper bound: 132.8428891
time: 5.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428982, upper bound: 132.8428891
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428982, upper bound: 132.8428891
time: 5.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428903, upper bound: 132.8428973
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428903, upper bound: 132.8428973
time: 6.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428903, upper bound: 132.8428973
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428903, upper bound: 132.8428973
time: 5.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428973, upper bound: 132.8428903
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428973, upper bound: 132.8428903
time: 5.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428973, upper bound: 132.8428903
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428973, upper bound: 132.8428903
time: 5.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428891, upper bound: 132.8428982
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428891, upper bound: 132.8428982
time: 5.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428891, upper bound: 132.8428982
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428891, upper bound: 132.8428982
time: 6.08 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 12.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.84
Output dim: 8, lower bound: -132.8428982, upper bound: 132.8428891
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.84
Output dim: 8, lower bound: -132.8428982, upper bound: 132.8428891
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.84
Output dim: 8, lower bound: -132.8428982, upper bound: 132.8428891
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.84
Output dim: 8, lower bound: -132.8428982, upper bound: 132.8428891
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.84
Output dim: 8, lower bound: -132.8428903, upper bound: 132.8428973
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.84
Output dim: 8, lower bound: -132.8428903, upper bound: 132.8428973
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.84
Output dim: 8, lower bound: -132.8428903, upper bound: 132.8428973
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.84
Output dim: 8, lower bound: -132.8428903, upper bound: 132.8428973
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.84
Output dim: 8, lower bound: -132.8428973, upper bound: 132.8428903
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.84
Output dim: 8, lower bound: -132.8428973, upper bound: 132.8428903
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.84
Output dim: 8, lower bound: -132.8428973, upper bound: 132.8428903
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.84
Output dim: 8, lower bound: -132.8428973, upper bound: 132.8428903
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.84
Output dim: 8, lower bound: -132.8428891, upper bound: 132.8428982
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.84
Output dim: 8, lower bound: -132.8428891, upper bound: 132.8428982
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.84
Output dim: 8, lower bound: -132.8428891, upper bound: 132.8428982
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.84
Output dim: 8, lower bound: -132.8428891, upper bound: 132.8428982
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=145.0349884033203
rel_dist={8: [-132.92657729879474, 132.92657729879477]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9204484, upper bound: 132.9203671
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9203672, upper bound: 132.9204485
time: 5.37 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.44 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.44
Output dim: 8, lower bound: -132.9204484, upper bound: 132.9203671
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.44
Output dim: 8, lower bound: -132.9203672, upper bound: 132.9204485

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9024333, upper bound: 132.9023252
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9023376, upper bound: 132.9024234
time: 4.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9024234, upper bound: 132.9023376
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9023252, upper bound: 132.9024333
time: 5.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 12.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.78
Output dim: 8, lower bound: -132.9024333, upper bound: 132.9023252
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.78
Output dim: 8, lower bound: -132.9023376, upper bound: 132.9024234
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.78
Output dim: 8, lower bound: -132.9024234, upper bound: 132.9023376
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.78
Output dim: 8, lower bound: -132.9023252, upper bound: 132.9024333

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8735138, upper bound: 132.8734244
time: 5.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8735138, upper bound: 132.8734244
time: 5.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734455, upper bound: 132.8734912
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734455, upper bound: 132.8734912
time: 5.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734912, upper bound: 132.8734455
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734912, upper bound: 132.8734455
time: 5.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734244, upper bound: 132.8735138
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734244, upper bound: 132.8735138
time: 4.96 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 11.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.24
Output dim: 8, lower bound: -132.8735138, upper bound: 132.8734244
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.24
Output dim: 8, lower bound: -132.8735138, upper bound: 132.8734244
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.24
Output dim: 8, lower bound: -132.8734455, upper bound: 132.8734912
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.24
Output dim: 8, lower bound: -132.8734455, upper bound: 132.8734912
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.24
Output dim: 8, lower bound: -132.8734912, upper bound: 132.8734455
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.24
Output dim: 8, lower bound: -132.8734912, upper bound: 132.8734455
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.24
Output dim: 8, lower bound: -132.8734244, upper bound: 132.8735138
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.24
Output dim: 8, lower bound: -132.8734244, upper bound: 132.8735138

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429130, upper bound: 132.8429016
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429130, upper bound: 132.8429016
time: 6.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429130, upper bound: 132.8429016
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429130, upper bound: 132.8429016
time: 5.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429032, upper bound: 132.8429116
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429032, upper bound: 132.8429116
time: 5.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429032, upper bound: 132.8429116
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429032, upper bound: 132.8429116
time: 5.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429116, upper bound: 132.8429032
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429116, upper bound: 132.8429032
time: 6.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429116, upper bound: 132.8429032
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429116, upper bound: 132.8429032
time: 4.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429016, upper bound: 132.8429130
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429016, upper bound: 132.8429130
time: 5.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429016, upper bound: 132.8429130
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429016, upper bound: 132.8429130
time: 6.54 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.59
Output dim: 8, lower bound: -132.8429130, upper bound: 132.8429016
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.59
Output dim: 8, lower bound: -132.8429130, upper bound: 132.8429016
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.59
Output dim: 8, lower bound: -132.8429130, upper bound: 132.8429016
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.59
Output dim: 8, lower bound: -132.8429130, upper bound: 132.8429016
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.59
Output dim: 8, lower bound: -132.8429032, upper bound: 132.8429116
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.59
Output dim: 8, lower bound: -132.8429032, upper bound: 132.8429116
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.59
Output dim: 8, lower bound: -132.8429032, upper bound: 132.8429116
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.59
Output dim: 8, lower bound: -132.8429032, upper bound: 132.8429116
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.59
Output dim: 8, lower bound: -132.8429116, upper bound: 132.8429032
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.59
Output dim: 8, lower bound: -132.8429116, upper bound: 132.8429032
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.59
Output dim: 8, lower bound: -132.8429116, upper bound: 132.8429032
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.59
Output dim: 8, lower bound: -132.8429116, upper bound: 132.8429032
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.59
Output dim: 8, lower bound: -132.8429016, upper bound: 132.8429130
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.59
Output dim: 8, lower bound: -132.8429016, upper bound: 132.8429130
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.59
Output dim: 8, lower bound: -132.8429016, upper bound: 132.8429130
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.59
Output dim: 8, lower bound: -132.8429016, upper bound: 132.8429130
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=145.0349884033203
rel_dist={8: [-132.92662008300115, 132.92662007898963]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9204690, upper bound: 132.9203834
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9203834, upper bound: 132.9204690
time: 8.05 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.90 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.90
Output dim: 8, lower bound: -132.9204690, upper bound: 132.9203834
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.90
Output dim: 8, lower bound: -132.9203834, upper bound: 132.9204690

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9024582, upper bound: 132.9023398
time: 7.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9023536, upper bound: 132.9024472
time: 5.88 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9024472, upper bound: 132.9023536
time: 6.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9023398, upper bound: 132.9024582
time: 6.83 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.95 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.95
Output dim: 8, lower bound: -132.9024582, upper bound: 132.9023398
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.95
Output dim: 8, lower bound: -132.9023536, upper bound: 132.9024472
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.95
Output dim: 8, lower bound: -132.9024472, upper bound: 132.9023536
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.95
Output dim: 8, lower bound: -132.9023398, upper bound: 132.9024582

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8735283, upper bound: 132.8734323
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8735283, upper bound: 132.8734323
time: 6.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734535, upper bound: 132.8735044
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734535, upper bound: 132.8735044
time: 5.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8735044, upper bound: 132.8734535
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8735044, upper bound: 132.8734535
time: 6.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734323, upper bound: 132.8735283
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8734323, upper bound: 132.8735283
time: 5.57 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 12.02 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.02
Output dim: 8, lower bound: -132.8735283, upper bound: 132.8734323
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.02
Output dim: 8, lower bound: -132.8735283, upper bound: 132.8734323
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.02
Output dim: 8, lower bound: -132.8734535, upper bound: 132.8735044
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.02
Output dim: 8, lower bound: -132.8734535, upper bound: 132.8735044
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.02
Output dim: 8, lower bound: -132.8735044, upper bound: 132.8734535
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.02
Output dim: 8, lower bound: -132.8735044, upper bound: 132.8734535
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.02
Output dim: 8, lower bound: -132.8734323, upper bound: 132.8735283
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.02
Output dim: 8, lower bound: -132.8734323, upper bound: 132.8735283

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429203, upper bound: 132.8429078
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429203, upper bound: 132.8429078
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429203, upper bound: 132.8429078
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429203, upper bound: 132.8429078
time: 5.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429093, upper bound: 132.8429182
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429093, upper bound: 132.8429182
time: 5.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429093, upper bound: 132.8429182
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429093, upper bound: 132.8429182
time: 5.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429182, upper bound: 132.8429093
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429182, upper bound: 132.8429093
time: 5.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429182, upper bound: 132.8429093
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429182, upper bound: 132.8429093
time: 5.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429078, upper bound: 132.8429203
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429078, upper bound: 132.8429203
time: 6.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -70.6837006, 56.5846901, -70.6837006, 56.5846901, -127.2683868, 127.2683868
1: -60.8618927, 49.8032379, -60.8618927, 49.8032379, -110.6651306, 110.6651306
2: -78.7016602, 50.0718803, -78.7016602, 50.0718803, -128.7734833, 128.7734985
3: -83.6095352, 43.2728920, -83.6095352, 43.2728920, -126.8824310, 126.8824310
4: -78.0087433, 57.9687424, -78.0087433, 57.9687424, -135.9774628, 135.9774780
5: -69.3549957, 53.8359833, -69.3549957, 53.8359833, -123.1909790, 123.1909790
6: -64.2819290, 62.9590263, -64.2819290, 62.9590263, -127.2409515, 127.2409515
7: -70.6644669, 62.5212440, -70.6644669, 62.5212440, -133.1857147, 133.1857147
8: -90.5866013, 54.4484024, -90.5866013, 54.4484024, -145.0349884, 145.0349884
9: -63.7630730, 63.7096710, -63.7630730, 63.7096710, -127.4727478, 127.4727478

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429078, upper bound: 132.8429203
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429078, upper bound: 132.8429203
time: 6.81 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.32
Output dim: 8, lower bound: -132.8429203, upper bound: 132.8429078
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.32
Output dim: 8, lower bound: -132.8429203, upper bound: 132.8429078
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.32
Output dim: 8, lower bound: -132.8429203, upper bound: 132.8429078
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.32
Output dim: 8, lower bound: -132.8429203, upper bound: 132.8429078
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.32
Output dim: 8, lower bound: -132.8429093, upper bound: 132.8429182
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.32
Output dim: 8, lower bound: -132.8429093, upper bound: 132.8429182
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.32
Output dim: 8, lower bound: -132.8429093, upper bound: 132.8429182
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.32
Output dim: 8, lower bound: -132.8429093, upper bound: 132.8429182
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.32
Output dim: 8, lower bound: -132.8429182, upper bound: 132.8429093
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.32
Output dim: 8, lower bound: -132.8429182, upper bound: 132.8429093
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.32
Output dim: 8, lower bound: -132.8429182, upper bound: 132.8429093
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.32
Output dim: 8, lower bound: -132.8429182, upper bound: 132.8429093
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.32
Output dim: 8, lower bound: -132.8429078, upper bound: 132.8429203
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.32
Output dim: 8, lower bound: -132.8429078, upper bound: 132.8429203
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.32
Output dim: 8, lower bound: -132.8429078, upper bound: 132.8429203
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.32
Output dim: 8, lower bound: -132.8429078, upper bound: 132.8429203
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=145.0349884033203
rel_dist={8: [-132.92664065775688, 132.92664062232905]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 850.20 seconds
