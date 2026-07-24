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
execution time: IAR + LP analysis = 1.14 + 10.90 = 12.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -132.9266407, upper bound: 132.9266406


# Binary Search by BASE starts (time budget: 2687.96 seconds, max iter: 100)

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
Binary search time: 35.34 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2652.62 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8990507, upper bound: 132.8990506
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8990507, upper bound: 132.8990506
time: 5.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.67
Output dim: 8, lower bound: -132.8990507, upper bound: 132.8990506
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.67
Output dim: 8, lower bound: -132.8990507, upper bound: 132.8990506

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568719
time: 6.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568719
time: 7.16 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8990507, upper bound: 132.8990403
time: 7.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8990403, upper bound: 132.8990506
time: 6.87 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.88
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568719
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.88
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568719
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.88
Output dim: 8, lower bound: -132.8990507, upper bound: 132.8990403
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.88
Output dim: 8, lower bound: -132.8990403, upper bound: 132.8990506

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568719
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568719
time: 8.31 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568713, upper bound: 132.8568719
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568713
time: 6.69 seconds

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8508259, upper bound: 132.8508244
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8508259, upper bound: 132.8508244
time: 5.30 seconds

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

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8990403, upper bound: 132.8990471
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8990381, upper bound: 132.8990506
time: 5.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.97
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568719
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.97
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568719
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.97
Output dim: 8, lower bound: -132.8568713, upper bound: 132.8568719
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.97
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568713
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.97
Output dim: 8, lower bound: -132.8508259, upper bound: 132.8508244
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.97
Output dim: 8, lower bound: -132.8508259, upper bound: 132.8508244
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.97
Output dim: 8, lower bound: -132.8990403, upper bound: 132.8990471
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.97
Output dim: 8, lower bound: -132.8990381, upper bound: 132.8990506

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568665
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568665, upper bound: 132.8568719
time: 6.71 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568665
time: 7.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568665, upper bound: 132.8568719
time: 5.80 seconds

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
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8418303, upper bound: 132.8418244
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8418244, upper bound: 132.8418304
time: 6.02 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8349806, upper bound: 132.8349639
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8349635, upper bound: 132.8349800
time: 5.18 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8394868, upper bound: 132.8394767
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8394767, upper bound: 132.8394862
time: 5.13 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8471222, upper bound: 132.8471141
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8471222, upper bound: 132.8471141
time: 6.11 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8990403, upper bound: 132.8990236
time: 8.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8990135, upper bound: 132.8990471
time: 7.22 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8762578, upper bound: 132.8762521
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8762347, upper bound: 132.8762874
time: 6.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.61
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568665
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.61
Output dim: 8, lower bound: -132.8568665, upper bound: 132.8568719
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.61
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568665
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.61
Output dim: 8, lower bound: -132.8568665, upper bound: 132.8568719
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.61
Output dim: 8, lower bound: -132.8418303, upper bound: 132.8418244
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.61
Output dim: 8, lower bound: -132.8418244, upper bound: 132.8418304
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.61
Output dim: 8, lower bound: -132.8349806, upper bound: 132.8349639
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.61
Output dim: 8, lower bound: -132.8349635, upper bound: 132.8349800
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.61
Output dim: 8, lower bound: -132.8394868, upper bound: 132.8394767
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.61
Output dim: 8, lower bound: -132.8394767, upper bound: 132.8394862
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 13.61
Output dim: 8, lower bound: -132.8471222, upper bound: 132.8471141
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 13.61
Output dim: 8, lower bound: -132.8471222, upper bound: 132.8471141
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.61
Output dim: 8, lower bound: -132.8990403, upper bound: 132.8990236
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.61
Output dim: 8, lower bound: -132.8990135, upper bound: 132.8990471
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.61
Output dim: 8, lower bound: -132.8762578, upper bound: 132.8762521
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.61
Output dim: 8, lower bound: -132.8762347, upper bound: 132.8762874

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568713, upper bound: 132.8568664
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568665
time: 8.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568665, upper bound: 132.8568719
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568664, upper bound: 132.8568713
time: 6.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568629, upper bound: 132.8568665
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568544
time: 8.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8567666, upper bound: 132.8567721
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8567666, upper bound: 132.8567721
time: 6.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8985803, upper bound: 132.8985880
time: 7.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8985803, upper bound: 132.8985880
time: 10.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8945450, upper bound: 132.8945212
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8945194, upper bound: 132.8945598
time: 7.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8762575, upper bound: 132.8762521
time: 6.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8762578, upper bound: 132.8762513
time: 7.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8753274, upper bound: 132.8753805
time: 7.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8753274, upper bound: 132.8753805
time: 6.53 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 15.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 8, lower bound: -132.8568713, upper bound: 132.8568664
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568665
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 8, lower bound: -132.8568665, upper bound: 132.8568719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 8, lower bound: -132.8568664, upper bound: 132.8568713
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 8, lower bound: -132.8568629, upper bound: 132.8568665
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568544
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 8, lower bound: -132.8567666, upper bound: 132.8567721
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 8, lower bound: -132.8567666, upper bound: 132.8567721
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 8, lower bound: -132.8985803, upper bound: 132.8985880
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 8, lower bound: -132.8985803, upper bound: 132.8985880
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 8, lower bound: -132.8945450, upper bound: 132.8945212
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 8, lower bound: -132.8945194, upper bound: 132.8945598
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 8, lower bound: -132.8762575, upper bound: 132.8762521
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 8, lower bound: -132.8762578, upper bound: 132.8762513
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 8, lower bound: -132.8753274, upper bound: 132.8753805
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.05
Output dim: 8, lower bound: -132.8753274, upper bound: 132.8753805

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8567381, upper bound: 132.8567343
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8567381, upper bound: 132.8567343
time: 6.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8309157, upper bound: 132.8309137
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8309157, upper bound: 132.8309137
time: 7.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8480175, upper bound: 132.8480074
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8480076, upper bound: 132.8480236
time: 7.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568664, upper bound: 132.8568651
time: 6.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568655, upper bound: 132.8568713
time: 6.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8418229, upper bound: 132.8418244
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8418195, upper bound: 132.8418304
time: 8.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568544
time: 7.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8568693, upper bound: 132.8568544
time: 7.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8435147, upper bound: 132.8435180
time: 8.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8435096, upper bound: 132.8435224
time: 6.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8477621, upper bound: 132.8477631
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8477547, upper bound: 132.8477707
time: 5.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8984688, upper bound: 132.8984911
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8984854, upper bound: 132.8984856
time: 6.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8910091, upper bound: 132.8910084
time: 7.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8910091, upper bound: 132.8910084
time: 6.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8719429, upper bound: 132.8718696
time: 7.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8718789, upper bound: 132.8719483
time: 6.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8868318, upper bound: 132.8868395
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8868145, upper bound: 132.8868432
time: 7.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8725859, upper bound: 132.8725962
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8725859, upper bound: 132.8725962
time: 6.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8677838, upper bound: 132.8677793
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8677838, upper bound: 132.8677802
time: 6.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8753215, upper bound: 132.8753803
time: 8.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8753273, upper bound: 132.8753778
time: 6.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8672868, upper bound: 132.8673263
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8672875, upper bound: 132.8673250
time: 6.15 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 13.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8567381, upper bound: 132.8567343
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8567381, upper bound: 132.8567343
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8309157, upper bound: 132.8309137
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8309157, upper bound: 132.8309137
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8480175, upper bound: 132.8480074
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8480076, upper bound: 132.8480236
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8568664, upper bound: 132.8568651
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8568655, upper bound: 132.8568713
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8418229, upper bound: 132.8418244
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8418195, upper bound: 132.8418304
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568544
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8568693, upper bound: 132.8568544
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8435147, upper bound: 132.8435180
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8435096, upper bound: 132.8435224
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8477621, upper bound: 132.8477631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8477547, upper bound: 132.8477707
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8984688, upper bound: 132.8984911
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8984854, upper bound: 132.8984856
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8910091, upper bound: 132.8910084
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8910091, upper bound: 132.8910084
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8719429, upper bound: 132.8718696
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8718789, upper bound: 132.8719483
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8868318, upper bound: 132.8868395
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8868145, upper bound: 132.8868432
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8725859, upper bound: 132.8725962
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8725859, upper bound: 132.8725962
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8677838, upper bound: 132.8677793
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8677838, upper bound: 132.8677802
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8753215, upper bound: 132.8753803
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8753273, upper bound: 132.8753778
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8672868, upper bound: 132.8673263
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.73
Output dim: 8, lower bound: -132.8672875, upper bound: 132.8673250

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8349772, upper bound: 132.8349629
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8349637, upper bound: 132.8349779
time: 5.92 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 11.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 11.98
Output dim: 8, lower bound: -132.8349772, upper bound: 132.8349629
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 11.98
Output dim: 8, lower bound: -132.8349637, upper bound: 132.8349779
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8567381, upper bound: 132.8567343
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8480175, upper bound: 132.8480074
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8480076, upper bound: 132.8480236
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8568664, upper bound: 132.8568651
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8568655, upper bound: 132.8568713
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8568719, upper bound: 132.8568544
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8568693, upper bound: 132.8568544
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8477621, upper bound: 132.8477631
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8477547, upper bound: 132.8477707
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8984688, upper bound: 132.8984911
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8984854, upper bound: 132.8984856
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8910091, upper bound: 132.8910084
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8910091, upper bound: 132.8910084
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8719429, upper bound: 132.8718696
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8718789, upper bound: 132.8719483
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8868318, upper bound: 132.8868395
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8868145, upper bound: 132.8868432
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8725859, upper bound: 132.8725962
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8725859, upper bound: 132.8725962
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8677838, upper bound: 132.8677793
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8677838, upper bound: 132.8677802
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8753215, upper bound: 132.8753803
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8753273, upper bound: 132.8753778
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8672868, upper bound: 132.8673263
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.98
Output dim: 8, lower bound: -132.8672875, upper bound: 132.8673250
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=145.0349884033203
rel_dist={8: [-132.92649402057955, 132.9264940353584]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9171713, upper bound: 132.9171654
time: 7.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9171654, upper bound: 132.9171713
time: 8.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.38
Output dim: 8, lower bound: -132.9171713, upper bound: 132.9171654
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.38
Output dim: 8, lower bound: -132.9171654, upper bound: 132.9171713

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8916474, upper bound: 132.8916384
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8916474, upper bound: 132.8916384
time: 6.40 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9165389, upper bound: 132.9165505
time: 9.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9165389, upper bound: 132.9165505
time: 10.98 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.88
Output dim: 8, lower bound: -132.8916474, upper bound: 132.8916384
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.88
Output dim: 8, lower bound: -132.8916474, upper bound: 132.8916384
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.88
Output dim: 8, lower bound: -132.9165389, upper bound: 132.9165505
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.88
Output dim: 8, lower bound: -132.9165389, upper bound: 132.9165505

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8706488, upper bound: 132.8706358
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8706365, upper bound: 132.8706468
time: 8.30 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8902251, upper bound: 132.8902143
time: 8.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8902251, upper bound: 132.8902143
time: 10.04 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8786440, upper bound: 132.8786440
time: 9.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8786440, upper bound: 132.8786440
time: 9.31 seconds

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
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9158961, upper bound: 132.9159042
time: 12.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9158961, upper bound: 132.9159037
time: 16.91 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 31.01 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 31.01
Output dim: 8, lower bound: -132.8706488, upper bound: 132.8706358
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 31.01
Output dim: 8, lower bound: -132.8706365, upper bound: 132.8706468
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 31.01
Output dim: 8, lower bound: -132.8902251, upper bound: 132.8902143
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 31.01
Output dim: 8, lower bound: -132.8902251, upper bound: 132.8902143
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 31.01
Output dim: 8, lower bound: -132.8786440, upper bound: 132.8786440
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 31.01
Output dim: 8, lower bound: -132.8786440, upper bound: 132.8786440
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 31.01
Output dim: 8, lower bound: -132.9158961, upper bound: 132.9159042
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 31.01
Output dim: 8, lower bound: -132.9158961, upper bound: 132.9159037

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
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8696617, upper bound: 132.8696471
time: 8.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8696617, upper bound: 132.8696471
time: 7.40 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8615078, upper bound: 132.8615283
time: 6.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8615078, upper bound: 132.8615283
time: 8.05 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8829732, upper bound: 132.8829663
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8829737, upper bound: 132.8829663
time: 7.51 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8726836, upper bound: 132.8726651
time: 7.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8726836, upper bound: 132.8726651
time: 9.90 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8786226, upper bound: 132.8786268
time: 8.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8786262, upper bound: 132.8786234
time: 8.08 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8777240, upper bound: 132.8777285
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8777240, upper bound: 132.8777285
time: 7.39 seconds

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
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8782469, upper bound: 132.8782488
time: 6.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8782469, upper bound: 132.8782488
time: 6.89 seconds

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9101186, upper bound: 132.9100912
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9100808, upper bound: 132.9101315
time: 9.73 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 18.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 8, lower bound: -132.8696617, upper bound: 132.8696471
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 8, lower bound: -132.8696617, upper bound: 132.8696471
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 8, lower bound: -132.8615078, upper bound: 132.8615283
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 8, lower bound: -132.8615078, upper bound: 132.8615283
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 8, lower bound: -132.8829732, upper bound: 132.8829663
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 8, lower bound: -132.8829737, upper bound: 132.8829663
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 8, lower bound: -132.8726836, upper bound: 132.8726651
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 8, lower bound: -132.8726836, upper bound: 132.8726651
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 8, lower bound: -132.8786226, upper bound: 132.8786268
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 8, lower bound: -132.8786262, upper bound: 132.8786234
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 8, lower bound: -132.8777240, upper bound: 132.8777285
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 8, lower bound: -132.8777240, upper bound: 132.8777285
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 8, lower bound: -132.8782469, upper bound: 132.8782488
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 8, lower bound: -132.8782469, upper bound: 132.8782488
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 8, lower bound: -132.9101186, upper bound: 132.9100912
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.27
Output dim: 8, lower bound: -132.9100808, upper bound: 132.9101315

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8527847, upper bound: 132.8527739
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8527847, upper bound: 132.8527739
time: 6.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8696587, upper bound: 132.8696471
time: 9.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8696617, upper bound: 132.8696451
time: 9.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8615078, upper bound: 132.8615249
time: 7.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8615070, upper bound: 132.8615283
time: 10.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8275058, upper bound: 132.8275253
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8275058, upper bound: 132.8275253
time: 7.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8745532, upper bound: 132.8745414
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8745327, upper bound: 132.8745589
time: 5.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8808617, upper bound: 132.8808520
time: 7.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8808617, upper bound: 132.8808520
time: 8.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8687217, upper bound: 132.8686869
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8686995, upper bound: 132.8687053
time: 7.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8726697, upper bound: 132.8726502
time: 8.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8726663, upper bound: 132.8726506
time: 7.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8673711, upper bound: 132.8673697
time: 9.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8673604, upper bound: 132.8673768
time: 7.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8777240, upper bound: 132.8777285
time: 6.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8777240, upper bound: 132.8777285
time: 8.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8613824, upper bound: 132.8613930
time: 8.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8613806, upper bound: 132.8613931
time: 7.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8719323, upper bound: 132.8719361
time: 6.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8719292, upper bound: 132.8719372
time: 10.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8757127, upper bound: 132.8757148
time: 8.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8757127, upper bound: 132.8757148
time: 8.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8782469, upper bound: 132.8782465
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8782415, upper bound: 132.8782488
time: 7.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9101186, upper bound: 132.9100898
time: 8.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9101183, upper bound: 132.9100912
time: 6.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8933814, upper bound: 132.8934077
time: 8.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8933814, upper bound: 132.8934077
time: 10.53 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8527847, upper bound: 132.8527739
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8527847, upper bound: 132.8527739
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8696587, upper bound: 132.8696471
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8696617, upper bound: 132.8696451
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8615078, upper bound: 132.8615249
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8615070, upper bound: 132.8615283
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8275058, upper bound: 132.8275253
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8275058, upper bound: 132.8275253
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8745532, upper bound: 132.8745414
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8745327, upper bound: 132.8745589
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8808617, upper bound: 132.8808520
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8808617, upper bound: 132.8808520
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8687217, upper bound: 132.8686869
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8686995, upper bound: 132.8687053
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8726697, upper bound: 132.8726502
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8726663, upper bound: 132.8726506
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8673711, upper bound: 132.8673697
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8673604, upper bound: 132.8673768
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8777240, upper bound: 132.8777285
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8777240, upper bound: 132.8777285
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8613824, upper bound: 132.8613930
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8613806, upper bound: 132.8613931
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8719323, upper bound: 132.8719361
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8719292, upper bound: 132.8719372
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8757127, upper bound: 132.8757148
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8757127, upper bound: 132.8757148
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8782469, upper bound: 132.8782465
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8782415, upper bound: 132.8782488
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.9101186, upper bound: 132.9100898
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.9101183, upper bound: 132.9100912
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8933814, upper bound: 132.8934077
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.23
Output dim: 8, lower bound: -132.8933814, upper bound: 132.8934077

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8527847, upper bound: 132.8527739
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8527847, upper bound: 132.8527739
time: 5.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8340817, upper bound: 132.8340598
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8340810, upper bound: 132.8340602
time: 10.82 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 20.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 20.06
Output dim: 8, lower bound: -132.8527847, upper bound: 132.8527739
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 20.06
Output dim: 8, lower bound: -132.8527847, upper bound: 132.8527739
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 20.06
Output dim: 8, lower bound: -132.8340817, upper bound: 132.8340598
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 20.06
Output dim: 8, lower bound: -132.8340810, upper bound: 132.8340602
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8696587, upper bound: 132.8696471
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8696617, upper bound: 132.8696451
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8615078, upper bound: 132.8615249
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8615070, upper bound: 132.8615283
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8745532, upper bound: 132.8745414
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8745327, upper bound: 132.8745589
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8808617, upper bound: 132.8808520
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8808617, upper bound: 132.8808520
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8687217, upper bound: 132.8686869
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8686995, upper bound: 132.8687053
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8726697, upper bound: 132.8726502
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8726663, upper bound: 132.8726506
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8673711, upper bound: 132.8673697
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8673604, upper bound: 132.8673768
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8777240, upper bound: 132.8777285
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8777240, upper bound: 132.8777285
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8613824, upper bound: 132.8613930
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8613806, upper bound: 132.8613931
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8719323, upper bound: 132.8719361
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8719292, upper bound: 132.8719372
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8757127, upper bound: 132.8757148
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8757127, upper bound: 132.8757148
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8782469, upper bound: 132.8782465
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8782415, upper bound: 132.8782488
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.9101186, upper bound: 132.9100898
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.9101183, upper bound: 132.9100912
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8933814, upper bound: 132.8934077
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.06
Output dim: 8, lower bound: -132.8933814, upper bound: 132.8934077
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=145.0349884033203
rel_dist={8: [-132.92639238641755, 132.92639237568028]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9088300, upper bound: 132.9088180
time: 12.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9088180, upper bound: 132.9088300
time: 7.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 20.20 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 20.20
Output dim: 8, lower bound: -132.9088300, upper bound: 132.9088180
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 20.20
Output dim: 8, lower bound: -132.9088180, upper bound: 132.9088300

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9088300, upper bound: 132.9088169
time: 10.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9088169, upper bound: 132.9088180
time: 11.44 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9033223, upper bound: 132.9033267
time: 12.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9033223, upper bound: 132.9033352
time: 8.00 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.49 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.49
Output dim: 8, lower bound: -132.9088300, upper bound: 132.9088169
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.49
Output dim: 8, lower bound: -132.9088169, upper bound: 132.9088180
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.49
Output dim: 8, lower bound: -132.9033223, upper bound: 132.9033267
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.49
Output dim: 8, lower bound: -132.9033223, upper bound: 132.9033352

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

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8842991, upper bound: 132.8842989
time: 10.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8843001, upper bound: 132.8842989
time: 11.76 seconds

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
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9041862, upper bound: 132.9041872
time: 8.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9041862, upper bound: 132.9041963
time: 8.63 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8943026, upper bound: 132.8943093
time: 8.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8943026, upper bound: 132.8943100
time: 9.58 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9033223, upper bound: 132.9033344
time: 9.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9033221, upper bound: 132.9033352
time: 11.18 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.47 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.47
Output dim: 8, lower bound: -132.8842991, upper bound: 132.8842989
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.47
Output dim: 8, lower bound: -132.8843001, upper bound: 132.8842989
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.47
Output dim: 8, lower bound: -132.9041862, upper bound: 132.9041872
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.47
Output dim: 8, lower bound: -132.9041862, upper bound: 132.9041963
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.47
Output dim: 8, lower bound: -132.8943026, upper bound: 132.8943093
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.47
Output dim: 8, lower bound: -132.8943026, upper bound: 132.8943100
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.47
Output dim: 8, lower bound: -132.9033223, upper bound: 132.9033344
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.47
Output dim: 8, lower bound: -132.9033221, upper bound: 132.9033352

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8726567, upper bound: 132.8726528
time: 9.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8726542, upper bound: 132.8726564
time: 9.83 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8842473, upper bound: 132.8842455
time: 8.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8842455, upper bound: 132.8842458
time: 10.08 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8625151, upper bound: 132.8625067
time: 8.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8625151, upper bound: 132.8625067
time: 8.62 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 216

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8614185, upper bound: 132.8614193
time: 7.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8614185, upper bound: 132.8614193
time: 7.80 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8762515, upper bound: 132.8762564
time: 9.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8762513, upper bound: 132.8762564
time: 8.46 seconds

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

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8717337, upper bound: 132.8717405
time: 8.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8717337, upper bound: 132.8717405
time: 8.34 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9023113, upper bound: 132.9023247
time: 8.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9023113, upper bound: 132.9023246
time: 9.39 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8848236, upper bound: 132.8848379
time: 8.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8848236, upper bound: 132.8848379
time: 8.81 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 18.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.45
Output dim: 8, lower bound: -132.8726567, upper bound: 132.8726528
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.45
Output dim: 8, lower bound: -132.8726542, upper bound: 132.8726564
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.45
Output dim: 8, lower bound: -132.8842473, upper bound: 132.8842455
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.45
Output dim: 8, lower bound: -132.8842455, upper bound: 132.8842458
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.45
Output dim: 8, lower bound: -132.8625151, upper bound: 132.8625067
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.45
Output dim: 8, lower bound: -132.8625151, upper bound: 132.8625067
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.45
Output dim: 8, lower bound: -132.8614185, upper bound: 132.8614193
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.45
Output dim: 8, lower bound: -132.8614185, upper bound: 132.8614193
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.45
Output dim: 8, lower bound: -132.8762515, upper bound: 132.8762564
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.45
Output dim: 8, lower bound: -132.8762513, upper bound: 132.8762564
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.45
Output dim: 8, lower bound: -132.8717337, upper bound: 132.8717405
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.45
Output dim: 8, lower bound: -132.8717337, upper bound: 132.8717405
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.45
Output dim: 8, lower bound: -132.9023113, upper bound: 132.9023247
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.45
Output dim: 8, lower bound: -132.9023113, upper bound: 132.9023246
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 18.45
Output dim: 8, lower bound: -132.8848236, upper bound: 132.8848379
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 18.45
Output dim: 8, lower bound: -132.8848236, upper bound: 132.8848379

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8682054, upper bound: 132.8682037
time: 9.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8682132, upper bound: 132.8682039
time: 8.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8423851, upper bound: 132.8423935
time: 9.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8423851, upper bound: 132.8423935
time: 7.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8842195, upper bound: 132.8842189
time: 7.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8842201, upper bound: 132.8842181
time: 7.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 216

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8842453, upper bound: 132.8842458
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8842455, upper bound: 132.8842457
time: 9.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8570762, upper bound: 132.8570703
time: 7.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8570756, upper bound: 132.8570730
time: 7.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8556578, upper bound: 132.8556528
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8556557, upper bound: 132.8556547
time: 8.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8426943, upper bound: 132.8426914
time: 8.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8426942, upper bound: 132.8426923
time: 7.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8599377, upper bound: 132.8599390
time: 8.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8599377, upper bound: 132.8599390
time: 8.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8762680, upper bound: 132.8762516
time: 9.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8762468, upper bound: 132.8762564
time: 10.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8749626, upper bound: 132.8749568
time: 8.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8749626, upper bound: 132.8749568
time: 8.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8550845, upper bound: 132.8550898
time: 8.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8550875, upper bound: 132.8550915
time: 8.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 134

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8717299, upper bound: 132.8717394
time: 7.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8717312, upper bound: 132.8717404
time: 7.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8805556, upper bound: 132.8805613
time: 8.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8805556, upper bound: 132.8805613
time: 10.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8783895, upper bound: 132.8783983
time: 10.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8783889, upper bound: 132.8783983
time: 8.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8845670, upper bound: 132.8845793
time: 9.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8845670, upper bound: 132.8845793
time: 10.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8732869, upper bound: 132.8732964
time: 8.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8732838, upper bound: 132.8733032
time: 11.07 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8682054, upper bound: 132.8682037
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8682132, upper bound: 132.8682039
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8423851, upper bound: 132.8423935
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8423851, upper bound: 132.8423935
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8842195, upper bound: 132.8842189
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8842201, upper bound: 132.8842181
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8842453, upper bound: 132.8842458
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8842455, upper bound: 132.8842457
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8570762, upper bound: 132.8570703
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8570756, upper bound: 132.8570730
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8556578, upper bound: 132.8556528
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8556557, upper bound: 132.8556547
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8426943, upper bound: 132.8426914
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8426942, upper bound: 132.8426923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8599377, upper bound: 132.8599390
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8599377, upper bound: 132.8599390
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8762680, upper bound: 132.8762516
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8762468, upper bound: 132.8762564
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8749626, upper bound: 132.8749568
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8749626, upper bound: 132.8749568
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8550845, upper bound: 132.8550898
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8550875, upper bound: 132.8550915
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8717299, upper bound: 132.8717394
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8717312, upper bound: 132.8717404
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8805556, upper bound: 132.8805613
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8805556, upper bound: 132.8805613
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8783895, upper bound: 132.8783983
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8783889, upper bound: 132.8783983
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8845670, upper bound: 132.8845793
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8845670, upper bound: 132.8845793
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8732869, upper bound: 132.8732964
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.51
Output dim: 8, lower bound: -132.8732838, upper bound: 132.8733032
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=145.0349884033203
rel_dist={8: [-132.9262935075506, 132.92629350749155]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1822.40 seconds
