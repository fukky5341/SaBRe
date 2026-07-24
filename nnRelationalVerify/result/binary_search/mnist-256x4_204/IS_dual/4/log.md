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
execution time: IAR + LP analysis = 1.10 + 10.46 = 11.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -132.9266407, upper bound: 132.9266406


# Binary Search by BASE starts (time budget: 2688.44 seconds, max iter: 100)

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
Binary search time: 34.44 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2654.00 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9200525, upper bound: 132.9191463
time: 5.84 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9203334, upper bound: 132.9203334
time: 9.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.51 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.51
Output dim: 8, lower bound: -132.9200525, upper bound: 132.9191463
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.51
Output dim: 8, lower bound: -132.9203334, upper bound: 132.9203334

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -65.2420654, 52.2499275, -70.6837006, 56.5846901, -121.8267517, 122.9336243
1: -56.2837639, 45.9009018, -60.8618927, 49.8032379, -106.0870056, 106.7627792
2: -72.7158813, 46.1605721, -78.7016602, 50.0718803, -122.7877426, 124.8622284
3: -77.1514893, 39.8565636, -83.6095352, 43.2728920, -120.4243774, 123.4660950
4: -72.1226959, 53.4341507, -78.0087433, 57.9687424, -130.0914307, 131.4428864
5: -64.0869293, 49.7719688, -69.3549957, 53.8359833, -117.9229126, 119.1269684
6: -59.2973900, 58.0655937, -64.2819290, 62.9590263, -122.2564087, 122.3475189
7: -65.1676102, 57.9393730, -70.6644669, 62.5212440, -127.6888580, 128.6038361
8: -84.0978088, 49.9053078, -90.5866013, 54.4484024, -138.5462036, 140.4919128
9: -58.7354584, 58.7163086, -63.7630730, 63.7096710, -122.4451141, 122.4793777

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9190575, upper bound: 132.9190575
time: 6.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9190575, upper bound: 132.9191201
time: 5.89 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -69.5974808, 55.6752396, -69.7686539, 55.8564606, -125.4539032, 125.4438934
1: -60.0971375, 48.9616241, -60.0911407, 49.1485863, -109.2457275, 109.0527649
2: -77.6256561, 49.2082863, -77.6937561, 49.4140129, -127.0396652, 126.9020386
3: -82.4195938, 42.4158249, -82.5280991, 42.7012405, -125.1208344, 124.9439011
4: -77.0379486, 56.9671783, -77.0201721, 57.2075691, -134.2455139, 133.9873505
5: -68.3729553, 53.0688896, -68.4698105, 53.1537437, -121.5266876, 121.5386963
6: -63.2544403, 61.9612122, -63.4423370, 62.1385727, -125.3930130, 125.4035416
7: -69.5783997, 61.8703766, -69.7409058, 61.7500877, -131.3284912, 131.6112823
8: -89.8109360, 52.9954147, -89.4902649, 53.6876526, -143.4985962, 142.4856873
9: -62.6431046, 62.6076050, -62.9195900, 62.8733673, -125.5164719, 125.5271912

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9191201, upper bound: 132.9200506
time: 6.45 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9191201, upper bound: 132.9203334
time: 9.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.90 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.90
Output dim: 8, lower bound: -132.9190575, upper bound: 132.9190575
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.90
Output dim: 8, lower bound: -132.9190575, upper bound: 132.9191201
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.90
Output dim: 8, lower bound: -132.9191201, upper bound: 132.9200506
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 16.90
Output dim: 8, lower bound: -132.9191201, upper bound: 132.9203334

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -65.2420654, 52.2499275, -65.2420654, 52.2499275, -117.4919891, 117.4919891
1: -56.2837639, 45.9009018, -56.2837639, 45.9009018, -102.1846619, 102.1846619
2: -72.7158813, 46.1605721, -72.7158813, 46.1605721, -118.8764496, 118.8764496
3: -77.1514893, 39.8565636, -77.1514893, 39.8565636, -117.0080338, 117.0080414
4: -72.1226959, 53.4341507, -72.1226959, 53.4341507, -125.5568390, 125.5568237
5: -64.0869293, 49.7719688, -64.0869293, 49.7719688, -113.8589020, 113.8589020
6: -59.2973900, 58.0655937, -59.2973900, 58.0655937, -117.3629837, 117.3629837
7: -65.1676102, 57.9393730, -65.1676102, 57.9393730, -123.1069794, 123.1069794
8: -84.0978088, 49.9053078, -84.0978088, 49.9053078, -134.0031128, 134.0031128
9: -58.7354584, 58.7163086, -58.7354584, 58.7163086, -117.4517593, 117.4517517

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8995290, upper bound: 132.8917002
time: 10.43 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9191117, upper bound: 132.9190976
time: 6.73 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -65.2420654, 52.2499275, -69.5974808, 55.6752396, -120.9173050, 121.8473969
1: -56.2837639, 45.9009018, -60.0971375, 48.9616241, -105.2453918, 105.9980316
2: -72.7158813, 46.1605721, -77.6256561, 49.2082863, -121.9241638, 123.7862244
3: -77.1514893, 39.8565636, -82.4195938, 42.4158249, -119.5673065, 122.2761536
4: -72.1226959, 53.4341507, -77.0379486, 56.9671783, -129.0898743, 130.4721069
5: -64.0869293, 49.7719688, -68.3729553, 53.0688896, -117.1558228, 118.1449280
6: -59.2973900, 58.0655937, -63.2544403, 61.9612122, -121.2585907, 121.3200378
7: -65.1676102, 57.9393730, -69.5783997, 61.8703766, -127.0379868, 127.5177765
8: -84.0978088, 49.9053078, -89.8109360, 52.9954147, -137.0932312, 139.7162476
9: -58.7354584, 58.7163086, -62.6431046, 62.6076050, -121.3430634, 121.3594131

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8995290, upper bound: 132.8917002
time: 10.97 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9191117, upper bound: 132.9191463
time: 7.17 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -69.5974808, 55.6752396, -65.2420654, 52.2499275, -121.8474121, 120.9173050
1: -60.0971375, 48.9616241, -56.2837639, 45.9009018, -105.9980316, 105.2453918
2: -77.6256561, 49.2082863, -72.7158813, 46.1605721, -123.7862244, 121.9241562
3: -82.4195938, 42.4158249, -77.1514893, 39.8565636, -122.2761536, 119.5673065
4: -77.0379486, 56.9671783, -72.1226959, 53.4341507, -130.4721069, 129.0898590
5: -68.3729553, 53.0688896, -64.0869293, 49.7719688, -118.1449280, 117.1558228
6: -63.2544403, 61.9612122, -59.2973900, 58.0655937, -121.3200378, 121.2585907
7: -69.5783997, 61.8703766, -65.1676102, 57.9393730, -127.5177765, 127.0379868
8: -89.8109360, 52.9954147, -84.0978088, 49.9053078, -139.7162476, 137.0932159
9: -62.6431046, 62.6076050, -58.7354584, 58.7163086, -121.3594131, 121.3430634

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8820140, upper bound: 132.8927059
time: 9.03 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9191201, upper bound: 132.9200506
time: 6.50 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -69.5974808, 55.6752396, -69.5974808, 55.6752396, -125.2727051, 125.2727051
1: -60.0971375, 48.9616241, -60.0971375, 48.9616241, -109.0587616, 109.0587616
2: -77.6256561, 49.2082863, -77.6256561, 49.2082863, -126.8339386, 126.8339310
3: -82.4195938, 42.4158249, -82.4195938, 42.4158249, -124.8354034, 124.8354187
4: -77.0379486, 56.9671783, -77.0379486, 56.9671783, -134.0051270, 134.0051270
5: -68.3729553, 53.0688896, -68.3729553, 53.0688896, -121.4418488, 121.4418488
6: -63.2544403, 61.9612122, -63.2544403, 61.9612122, -125.2156525, 125.2156525
7: -69.5783997, 61.8703766, -69.5783997, 61.8703766, -131.4487762, 131.4487762
8: -89.8109360, 52.9954147, -89.8109360, 52.9954147, -142.8063507, 142.8063507
9: -62.6431046, 62.6076050, -62.6431046, 62.6076050, -125.2507019, 125.2507019

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8919489, upper bound: 132.8819679
time: 7.56 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9191201, upper bound: 132.9203334
time: 6.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 15.16 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.16
Output dim: 8, lower bound: -132.8995290, upper bound: 132.8917002
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.16
Output dim: 8, lower bound: -132.9191117, upper bound: 132.9190976
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.16
Output dim: 8, lower bound: -132.8995290, upper bound: 132.8917002
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.16
Output dim: 8, lower bound: -132.9191117, upper bound: 132.9191463
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 15.16
Output dim: 8, lower bound: -132.8820140, upper bound: 132.8927059
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 15.16
Output dim: 8, lower bound: -132.9191201, upper bound: 132.9200506
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.16
Output dim: 8, lower bound: -132.8919489, upper bound: 132.8819679
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.16
Output dim: 8, lower bound: -132.9191201, upper bound: 132.9203334

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -70.2937317, 56.2881546, -64.9590988, 52.0267563, -122.3204880, 121.2472534
1: -60.6407623, 49.4441299, -56.0392685, 45.7026367, -106.3433914, 105.4833984
2: -78.3662567, 49.7555923, -72.4003601, 45.9614944, -124.3277512, 122.1559448
3: -83.0576248, 42.8847656, -76.8157043, 39.6889954, -122.7466202, 119.7004547
4: -77.6926422, 57.6051140, -71.8099365, 53.2055740, -130.8982239, 129.4150391
5: -69.0405350, 53.5901566, -63.8123779, 49.5607033, -118.6012268, 117.4025269
6: -63.9047699, 62.6133537, -59.0410767, 57.8130035, -121.7177734, 121.6544189
7: -70.2367706, 62.3748817, -64.8836823, 57.6925240, -127.9292755, 127.2585602
8: -90.5085144, 53.8809052, -83.7391434, 49.6923752, -140.2008820, 137.6200562
9: -63.3077393, 63.3130951, -58.4812012, 58.4633865, -121.7711258, 121.7942963

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8627753, upper bound: 132.8628945
time: 8.91 seconds

## Relational analysis of IS_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8805084, upper bound: 132.8724684
time: 9.82 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8778846, upper bound: 132.8673227
time: 9.67 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -65.0691071, 52.1140366, -65.2420654, 52.2499275, -117.3190308, 117.3561020
1: -56.1351242, 45.7792931, -56.2837639, 45.9009018, -102.0360107, 102.0630569
2: -72.5241547, 46.0391197, -72.7158813, 46.1605721, -118.6847229, 118.7549973
3: -76.9457245, 39.7533722, -77.1514893, 39.8565636, -116.8022842, 116.9048615
4: -71.9329376, 53.2940903, -72.1226959, 53.4341507, -125.3670883, 125.4167786
5: -63.9201965, 49.6434708, -64.0869293, 49.7719688, -113.6921692, 113.7303848
6: -59.1408997, 57.9115448, -59.2973900, 58.0655937, -117.2064972, 117.2089386
7: -64.9947739, 57.7908134, -65.1676102, 57.9393730, -122.9341431, 122.9584198
8: -83.8812790, 49.7732430, -84.0978088, 49.9053078, -133.7865906, 133.8710480
9: -58.5801010, 58.5627518, -58.7354584, 58.7163086, -117.2964020, 117.2982025

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9004193, upper bound: 132.9010880
time: 7.87 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9019338, upper bound: 132.9019339
time: 8.25 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -70.2937317, 56.2881546, -69.3041992, 55.4440842, -125.7378082, 125.5923538
1: -60.6407623, 49.4441299, -59.8436127, 48.7560043, -109.3967590, 109.2877426
2: -78.3662567, 49.7555923, -77.2986298, 49.0018578, -127.3681183, 127.0542221
3: -83.0576248, 42.8847656, -82.0713348, 42.2418556, -125.2994843, 124.9561005
4: -77.6926422, 57.6051140, -76.7139435, 56.7300682, -134.4226837, 134.3190460
5: -69.0405350, 53.5901566, -68.0882797, 52.8499565, -121.8904877, 121.6784363
6: -63.9047699, 62.6133537, -62.9886017, 61.6993942, -125.6041565, 125.6019516
7: -70.2367706, 62.3748817, -69.2841721, 61.6146736, -131.8514404, 131.6590576
8: -90.5085144, 53.8809052, -89.4393616, 52.7745628, -143.2830658, 143.3202667
9: -63.3077393, 63.3130951, -62.3794556, 62.3452606, -125.6529999, 125.6925507

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8624841, upper bound: 132.8618923
time: 11.77 seconds

## Relational analysis of IS_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8694945, upper bound: 132.8652462
time: 9.15 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8573882, upper bound: 132.8478753
time: 8.71 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -65.0691071, 52.1140366, -69.5974808, 55.6752396, -120.7443466, 121.7115021
1: -56.1351242, 45.7792931, -60.0971375, 48.9616241, -105.0967407, 105.8764343
2: -72.5241547, 46.0391197, -77.6256561, 49.2082863, -121.7324371, 123.6647415
3: -76.9457245, 39.7533722, -82.4195938, 42.4158249, -119.3615494, 122.1729660
4: -71.9329376, 53.2940903, -77.0379486, 56.9671783, -128.9001160, 130.3320312
5: -63.9201965, 49.6434708, -68.3729553, 53.0688896, -116.9890900, 118.0164185
6: -59.1408997, 57.9115448, -63.2544403, 61.9612122, -121.1021118, 121.1659851
7: -64.9947739, 57.7908134, -69.5783997, 61.8703766, -126.8651505, 127.3691864
8: -83.8812790, 49.7732430, -89.8109360, 52.9954147, -136.8766937, 139.5841827
9: -58.5801010, 58.5627518, -62.6431046, 62.6076050, -121.1877060, 121.2058563

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9005429, upper bound: 132.9008825
time: 7.35 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9021602, upper bound: 132.9017321
time: 6.39 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -69.3041992, 55.4440842, -70.2937317, 56.2881546, -125.5923538, 125.7378159
1: -59.8436127, 48.7560043, -60.6407623, 49.4441299, -109.2877426, 109.3967590
2: -77.2986298, 49.0018578, -78.3662567, 49.7555923, -127.0542221, 127.3681107
3: -82.0713348, 42.2418556, -83.0576248, 42.8847656, -124.9561005, 125.2994843
4: -76.7139435, 56.7300682, -77.6926422, 57.6051140, -134.3190460, 134.4227142
5: -68.0882797, 52.8499565, -69.0405350, 53.5901566, -121.6784363, 121.8904877
6: -62.9886017, 61.6993942, -63.9047699, 62.6133537, -125.6019592, 125.6041565
7: -69.2841721, 61.6146736, -70.2367706, 62.3748817, -131.6590576, 131.8514404
8: -89.4393616, 52.7745628, -90.5085144, 53.8809052, -143.3202667, 143.2830658
9: -62.3794556, 62.3452606, -63.3077393, 63.3130951, -125.6925507, 125.6529999

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8618923, upper bound: 132.8624841
time: 8.26 seconds

## Relational analysis of IS_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8652462, upper bound: 132.8694945
time: 7.81 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8478753, upper bound: 132.8573882
time: 6.17 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -69.5974808, 55.6752396, -65.0691071, 52.1140366, -121.7115173, 120.7443466
1: -60.0971375, 48.9616241, -56.1351242, 45.7792931, -105.8764343, 105.0967407
2: -77.6256561, 49.2082863, -72.5241547, 46.0391197, -123.6647644, 121.7324371
3: -82.4195938, 42.4158249, -76.9457245, 39.7533722, -122.1729660, 119.3615417
4: -77.0379486, 56.9671783, -71.9329376, 53.2940903, -130.3320312, 128.9001160
5: -68.3729553, 53.0688896, -63.9201965, 49.6434708, -118.0164261, 116.9890823
6: -63.2544403, 61.9612122, -59.1408997, 57.9115448, -121.1659851, 121.1021118
7: -69.5783997, 61.8703766, -64.9947739, 57.7908134, -127.3692017, 126.8651505
8: -89.8109360, 52.9954147, -83.8812790, 49.7732430, -139.5841675, 136.8766785
9: -62.6431046, 62.6076050, -58.5801010, 58.5627518, -121.2058563, 121.1877060

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9008825, upper bound: 132.9005429
time: 9.41 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9017321, upper bound: 132.9021602
time: 6.26 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -74.5128021, 59.6170807, -69.3041992, 55.4440842, -129.9568634, 128.9212799
1: -64.3241501, 52.4080582, -59.8436127, 48.7560043, -113.0801544, 112.2516708
2: -83.1217422, 52.7119026, -77.2986298, 49.0018578, -132.1235962, 130.0105286
3: -88.1526566, 45.3759155, -82.0713348, 42.2418556, -130.3945160, 127.4472351
4: -82.4438019, 61.0340805, -76.7139435, 56.7300682, -139.1738586, 137.7480164
5: -73.1953583, 56.7821426, -68.0882797, 52.8499565, -126.0453186, 124.8704071
6: -67.7505951, 66.3912125, -62.9886017, 61.6993942, -129.4499817, 129.3798218
7: -74.5169373, 66.1709595, -69.2841721, 61.6146736, -136.1316071, 135.4551086
8: -96.0214157, 56.9146652, -89.4393616, 52.7745628, -148.7959595, 146.3540039
9: -67.1064224, 67.0939636, -62.3794556, 62.3452606, -129.4516754, 129.4734192

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8481449, upper bound: 132.8487702
time: 10.07 seconds

## Relational analysis of IS_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8676675, upper bound: 132.8638622
time: 10.19 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8919489, upper bound: 132.8819679
time: 8.58 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -69.4205017, 55.5362740, -69.5974808, 55.6752396, -125.0957336, 125.1337280
1: -59.9449081, 48.8372421, -60.0971375, 48.9616241, -108.9065170, 108.9343719
2: -77.4293976, 49.0839424, -77.6256561, 49.2082863, -126.6376801, 126.7095871
3: -82.2090836, 42.3102951, -82.4195938, 42.4158249, -124.6248779, 124.7298889
4: -76.8436890, 56.8238564, -77.0379486, 56.9671783, -133.8108673, 133.8618011
5: -68.2021790, 52.9374504, -68.3729553, 53.0688896, -121.2710571, 121.3104095
6: -63.0942955, 61.8035393, -63.2544403, 61.9612122, -125.0554810, 125.0579834
7: -69.4014893, 61.7182159, -69.5783997, 61.8703766, -131.2718658, 131.2966156
8: -89.5891571, 52.8604546, -89.8109360, 52.9954147, -142.5845642, 142.6713867
9: -62.4841614, 62.4503670, -62.6431046, 62.6076050, -125.0917511, 125.0934753

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9117277, upper bound: 132.9147302
time: 10.25 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9202833, upper bound: 132.9203334
time: 6.89 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.28 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -132.8805084, upper bound: 132.8724684
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -132.8778846, upper bound: 132.8673227
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -132.9004193, upper bound: 132.9010880
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -132.9019338, upper bound: 132.9019339
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -132.8694945, upper bound: 132.8652462
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -132.8573882, upper bound: 132.8478753
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -132.9005429, upper bound: 132.9008825
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -132.9021602, upper bound: 132.9017321
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -132.8652462, upper bound: 132.8694945
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -132.8478753, upper bound: 132.8573882
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -132.9008825, upper bound: 132.9005429
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -132.9017321, upper bound: 132.9021602
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -132.8676675, upper bound: 132.8638622
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -132.8919489, upper bound: 132.8819679
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -132.9117277, upper bound: 132.9147302
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -132.9202833, upper bound: 132.9203334

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -60.4500885, 48.3803444, -64.9590988, 52.0267563, -112.4768448, 113.3394394
1: -52.2669868, 42.4472656, -56.0392685, 45.7026367, -97.9696198, 98.4865341
2: -67.4159470, 42.7761040, -72.4003601, 45.9614944, -113.3774414, 115.1764679
3: -71.3466263, 36.8460350, -76.8157043, 39.6889954, -111.0356216, 113.6617355
4: -66.9204941, 49.4447021, -71.8099365, 53.2055740, -120.1260605, 121.2546387
5: -59.3458824, 46.1062317, -63.8123779, 49.5607033, -108.9065781, 109.9186096
6: -54.9207230, 53.7901344, -59.0410767, 57.8130035, -112.7337265, 112.8311920
7: -60.3241272, 53.8464127, -64.8836823, 57.6925240, -118.0166473, 118.7300873
8: -78.4259262, 45.9684906, -83.7391434, 49.6923752, -128.1183014, 129.7076416
9: -54.2160683, 54.3818626, -58.4812012, 58.4633865, -112.6794586, 112.8630447

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8760517, upper bound: 132.8670138
time: 7.06 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8760517, upper bound: 132.8673226
time: 7.96 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -63.6586761, 50.9038696, -62.5480118, 50.0946503, -113.7533264, 113.4518661
1: -55.1026993, 44.6667023, -53.9888687, 43.9878693, -99.0905685, 98.6555710
2: -71.0071869, 44.9982224, -69.7150345, 44.2525864, -115.2597733, 114.7132568
3: -75.2108612, 38.7084465, -73.9404984, 38.2128105, -113.4236755, 112.6489410
4: -70.5662842, 51.9933434, -69.1633301, 51.2125511, -121.7788391, 121.1566696
5: -62.5059013, 48.5679512, -61.4484291, 47.7328186, -110.2387238, 110.0163803
6: -57.8757706, 56.6230888, -56.8402290, 55.6494942, -113.5252686, 113.4633179
7: -63.5603676, 56.7973175, -62.4517822, 55.6070023, -119.1673737, 119.2490692
8: -82.7404785, 48.1421204, -80.7679977, 47.7588806, -130.4993591, 128.9101257
9: -57.0590363, 57.2112503, -56.2594910, 56.2789154, -113.3379517, 113.4707413

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8553948, upper bound: 132.8513870
time: 10.14 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8778846, upper bound: 132.8673227
time: 6.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -65.0691071, 52.1140366, -55.5774879, 44.4827690, -109.5518646, 107.6915283
1: -56.1351242, 45.7792931, -48.0620079, 39.0292854, -95.1644135, 93.8412933
2: -72.5241547, 46.0391197, -61.9686203, 39.3269501, -111.8511047, 108.0077286
3: -76.9457245, 39.7533722, -65.6207657, 33.9203949, -110.8661118, 105.3741302
4: -71.9329376, 53.2940903, -61.5357552, 45.4231720, -117.3561096, 114.8298492
5: -63.9201965, 49.6434708, -54.5577545, 42.4212570, -106.3414459, 104.2012024
6: -59.1408997, 57.9115448, -50.4784355, 49.4100761, -108.5509796, 108.3899841
7: -64.9947739, 57.7908134, -55.4245872, 49.5640755, -114.5588303, 113.2153931
8: -83.8812790, 49.7732430, -72.2206039, 42.1627083, -126.0439758, 121.9938354
9: -58.5801010, 58.5627518, -49.8079758, 49.9667969, -108.5468979, 108.3707275

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8991023, upper bound: 132.8991023
time: 6.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8991023, upper bound: 132.9008128
time: 9.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -62.6571503, 50.1812897, -58.7964249, 47.0061340, -109.6632843, 108.9776993
1: -54.0839806, 44.0639496, -50.9198990, 41.2602539, -95.3442383, 94.9838486
2: -69.8379059, 44.3295746, -65.5706024, 41.5586014, -111.3965073, 109.9001770
3: -74.0696106, 38.2766800, -69.5020905, 35.7885399, -109.8581543, 107.7787704
4: -69.2854691, 51.3003578, -65.1963577, 47.9783478, -117.2638168, 116.4967194
5: -61.5555000, 47.8149681, -57.7157669, 44.8917198, -106.4472198, 105.5307312
6: -56.9393883, 55.7472916, -53.4313278, 52.2572403, -109.1966248, 109.1786194
7: -62.5620995, 55.7045250, -58.6720810, 52.5250626, -115.0871506, 114.3765945
8: -80.9091492, 47.8391724, -76.5529938, 44.3412323, -125.2503815, 124.3921661
9: -56.3576164, 56.3774605, -52.6564674, 52.8031311, -109.1607513, 109.0339279

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8756468, upper bound: 132.8793655
time: 7.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8731086, upper bound: 132.8731086
time: 6.39 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -62.6625328, 50.1852951, -69.3041992, 55.4440842, -118.1066055, 119.4894943
1: -54.1614609, 44.0396461, -59.8436127, 48.7560043, -102.9174652, 103.8832550
2: -69.8593750, 44.2875977, -77.2986298, 49.0018578, -118.8612366, 121.5862274
3: -74.0365524, 38.1787415, -82.0713348, 42.2418556, -116.2784042, 120.2500687
4: -69.4043121, 51.2879181, -76.7139435, 56.7300682, -126.1343689, 128.0018616
5: -61.6186867, 47.8674011, -68.0882797, 52.8499565, -114.4686432, 115.9556656
6: -56.8740730, 55.7905884, -62.9886017, 61.6993942, -118.5734711, 118.7791748
7: -62.5371933, 55.8827782, -69.2841721, 61.6146736, -124.1518555, 125.1669464
8: -81.2626266, 47.5885162, -89.4393616, 52.7745628, -134.0371857, 137.0278778
9: -56.3457603, 56.4394989, -62.3794556, 62.3452606, -118.6910095, 118.8189545

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8366797, upper bound: 132.8342400
time: 6.72 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8694945, upper bound: 132.8652462
time: 6.49 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -58.7831421, 47.0155830, -65.6236115, 52.5065269, -111.2896500, 112.6391907
1: -51.0528831, 41.2081451, -56.7287903, 46.1428261, -97.1957092, 97.9369125
2: -65.5988388, 41.4397812, -73.2011642, 46.3744888, -111.9733276, 114.6409302
3: -69.4806671, 35.6489716, -77.7207260, 39.9760818, -109.4567490, 113.3696823
4: -65.4057388, 47.9551430, -72.7139206, 53.6856613, -119.0914001, 120.6690674
5: -57.8701668, 45.0018158, -64.5165787, 50.0933418, -107.9635086, 109.5183945
6: -53.2736206, 52.3429756, -59.6116333, 58.4082756, -111.6819000, 111.9545975
7: -58.6542091, 52.8323517, -65.5791245, 58.4912071, -117.1454163, 118.4114761
8: -77.0411072, 43.8385544, -84.9817810, 49.7392120, -126.7803192, 128.8203278
9: -52.6872749, 52.8540649, -59.0152473, 59.0212555, -111.7085266, 111.8693085

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8097105, upper bound: 132.8094270
time: 10.41 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8325576, upper bound: 132.8247904
time: 8.24 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8573882, upper bound: 132.8478753
time: 7.92 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8573882, upper bound: 132.8478753
time: 9.28 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -65.0691071, 52.1140366, -59.7804794, 47.7900887, -112.8591919, 111.8945160
1: -56.1351242, 45.7792931, -51.7556534, 41.9793549, -98.1144638, 97.5349426
2: -72.5241547, 46.0391197, -66.7171249, 42.2620277, -114.7861786, 112.7562332
3: -76.9457245, 39.7533722, -70.7146072, 36.3832741, -113.3289948, 110.4679794
4: -71.9329376, 53.2940903, -66.2887115, 48.8286781, -120.7616119, 119.5828018
5: -63.9201965, 49.6434708, -58.7035522, 45.6050606, -109.5252533, 108.3470154
6: -59.1408997, 57.9115448, -54.2974281, 53.1681366, -112.3090286, 112.2089691
7: -64.9947739, 57.7908134, -59.6878586, 53.3782234, -118.3729782, 117.4786606
8: -83.8812790, 49.7732430, -77.7729263, 45.1020164, -128.9832916, 127.5461578
9: -58.5801010, 58.5627518, -53.5734024, 53.7139206, -112.2940216, 112.1361389

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8992211, upper bound: 132.8988928
time: 7.13 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8992211, upper bound: 132.9006252
time: 6.63 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -62.6571503, 50.1812897, -63.0181198, 50.3360786, -112.9932098, 113.1994019
1: -54.0839806, 44.0639496, -54.6180954, 44.2214661, -98.3054504, 98.6820374
2: -69.8379059, 44.3295746, -70.3297348, 44.5051956, -114.3431015, 114.6593094
3: -74.0696106, 38.2766800, -74.6101074, 38.2674561, -112.3370514, 112.8867874
4: -69.2854691, 51.3003578, -69.9527359, 51.4075546, -120.6930084, 121.2530899
5: -61.5555000, 47.8149681, -61.8854828, 48.0801697, -109.6356659, 109.7004547
6: -56.9393883, 55.7472916, -57.2744293, 56.0269623, -112.9663544, 113.0217209
7: -62.5620995, 55.7045250, -62.9446182, 56.3383522, -118.9004364, 118.6491241
8: -80.9091492, 47.8391724, -82.0898514, 47.3319855, -128.2411346, 129.9290009
9: -56.3576164, 56.3774605, -56.4451141, 56.5664139, -112.9240265, 112.8225708

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8754002, upper bound: 132.8800517
time: 7.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8696820, upper bound: 132.8676043
time: 6.74 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -69.3041992, 55.4440842, -62.6625328, 50.1852951, -119.4894943, 118.1066055
1: -59.8436127, 48.7560043, -54.1614609, 44.0396461, -103.8832550, 102.9174652
2: -77.2986298, 49.0018578, -69.8593750, 44.2875977, -121.5862274, 118.8612366
3: -82.0713348, 42.2418556, -74.0365524, 38.1787415, -120.2500687, 116.2784119
4: -76.7139435, 56.7300682, -69.4043121, 51.2879181, -128.0018616, 126.1343689
5: -68.0882797, 52.8499565, -61.6186867, 47.8674011, -115.9556656, 114.4686432
6: -62.9886017, 61.6993942, -56.8740730, 55.7905884, -118.7791824, 118.5734711
7: -69.2841721, 61.6146736, -62.5371933, 55.8827782, -125.1669464, 124.1518555
8: -89.4393616, 52.7745628, -81.2626266, 47.5885162, -137.0278778, 134.0371857
9: -62.3794556, 62.3452606, -56.3457603, 56.4394989, -118.8189545, 118.6910095

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8342400, upper bound: 132.8366797
time: 11.08 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8652462, upper bound: 132.8694945
time: 9.81 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -65.6236115, 52.5065269, -58.7831421, 47.0155830, -112.6391907, 111.2896500
1: -56.7287903, 46.1428261, -51.0528831, 41.2081451, -97.9369125, 97.1957092
2: -73.2011642, 46.3744888, -65.5988388, 41.4397812, -114.6409302, 111.9733276
3: -77.7207260, 39.9760818, -69.4806671, 35.6489716, -113.3696823, 109.4567490
4: -72.7139206, 53.6856613, -65.4057388, 47.9551430, -120.6690674, 119.0914001
5: -64.5165787, 50.0933418, -57.8701668, 45.0018158, -109.5183868, 107.9635086
6: -59.6116333, 58.4082756, -53.2736206, 52.3429756, -111.9545898, 111.6819000
7: -65.5791245, 58.4912071, -58.6542091, 52.8323517, -118.4114761, 117.1454163
8: -84.9817810, 49.7392120, -77.0411072, 43.8385544, -128.8203430, 126.7803192
9: -59.0152473, 59.0212555, -52.6872749, 52.8540649, -111.8693085, 111.7085266

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8094270, upper bound: 132.8097104
time: 10.88 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8247904, upper bound: 132.8325576
time: 11.66 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of IS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8478753, upper bound: 132.8573882
time: 7.25 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8478753, upper bound: 132.8573882
time: 9.74 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -59.7804794, 47.7900887, -65.0691071, 52.1140366, -111.8945160, 112.8591919
1: -51.7556534, 41.9793549, -56.1351242, 45.7792931, -97.5349426, 98.1144714
2: -66.7171249, 42.2620277, -72.5241547, 46.0391197, -112.7562332, 114.7861786
3: -70.7146072, 36.3832741, -76.9457245, 39.7533722, -110.4679718, 113.3289948
4: -66.2887115, 48.8286781, -71.9329376, 53.2940903, -119.5828018, 120.7616043
5: -58.7035522, 45.6050606, -63.9201965, 49.6434708, -108.3470154, 109.5252533
6: -54.2974281, 53.1681366, -59.1408997, 57.9115448, -112.2089691, 112.3090286
7: -59.6878586, 53.3782234, -64.9947739, 57.7908134, -117.4786682, 118.3729935
8: -77.7729263, 45.1020164, -83.8812790, 49.7732430, -127.5461731, 128.9832916
9: -53.5734024, 53.7139206, -58.5801010, 58.5627518, -112.1361389, 112.2940216

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8988928, upper bound: 132.8992211
time: 7.27 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8988928, upper bound: 132.8995003
time: 6.86 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -63.0181198, 50.3360786, -62.6571503, 50.1812897, -113.1994019, 112.9932098
1: -54.6180954, 44.2214661, -54.0839806, 44.0639496, -98.6820450, 98.3054504
2: -70.3297348, 44.5051956, -69.8379059, 44.3295746, -114.6593018, 114.3431015
3: -74.6101074, 38.2674561, -74.0696106, 38.2766800, -112.8867874, 112.3370514
4: -69.9527359, 51.4075546, -69.2854691, 51.3003578, -121.2530823, 120.6930237
5: -61.8854828, 48.0801697, -61.5555000, 47.8149681, -109.7004547, 109.6356659
6: -57.2744293, 56.0269623, -56.9393883, 55.7472916, -113.0217209, 112.9663544
7: -62.9446182, 56.3383522, -62.5620995, 55.7045250, -118.6491394, 118.9004364
8: -82.0898514, 47.3319855, -80.9091492, 47.8391724, -129.9290009, 128.2411346
9: -56.4451141, 56.5664139, -56.3576164, 56.3774605, -112.8225708, 112.9240265

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8800517, upper bound: 132.8754002
time: 11.30 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8676043, upper bound: 132.8696820
time: 6.83 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 19.29 seconds
IS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8760517, upper bound: 132.8670138
IS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8760517, upper bound: 132.8673226
IS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8553948, upper bound: 132.8513870
IS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8778846, upper bound: 132.8673227
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8991023, upper bound: 132.8991023
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8991023, upper bound: 132.9008128
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8756468, upper bound: 132.8793655
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8731086, upper bound: 132.8731086
IS_A1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8366797, upper bound: 132.8342400
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8694945, upper bound: 132.8652462
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8573882, upper bound: 132.8478753
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8573882, upper bound: 132.8478753
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8992211, upper bound: 132.8988928
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8992211, upper bound: 132.9006252
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8754002, upper bound: 132.8800517
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8696820, upper bound: 132.8676043
IS_A2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8342400, upper bound: 132.8366797
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8652462, upper bound: 132.8694945
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8478753, upper bound: 132.8573882
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8478753, upper bound: 132.8573882
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8988928, upper bound: 132.8992211
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8988928, upper bound: 132.8995003
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8800517, upper bound: 132.8754002
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 19.29
Output dim: 8, lower bound: -132.8676043, upper bound: 132.8696820
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.29
Output dim: 8, lower bound: -132.8676675, upper bound: 132.8638622
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.29
Output dim: 8, lower bound: -132.8919489, upper bound: 132.8819679
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.29
Output dim: 8, lower bound: -132.9117277, upper bound: 132.9147302
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.29
Output dim: 8, lower bound: -132.9202833, upper bound: 132.9203334
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=145.0349884033203
rel_dist={8: [-132.92649402057955, 132.9264940353584]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9030011, upper bound: 132.8982915
time: 11.16 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9263924, upper bound: 132.9263924
time: 8.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 20.14 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 20.14
Output dim: 8, lower bound: -132.9030011, upper bound: 132.8982915
IS_A2, status: Status.UNKNOWN, split count: 1, time: 20.14
Output dim: 8, lower bound: -132.9263924, upper bound: 132.9263924

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -75.7929382, 60.6694870, -69.9614334, 56.0156708, -131.8085785, 130.6309204
1: -65.2676163, 53.3889885, -60.2387848, 49.2968178, -114.5644302, 113.6277695
2: -84.4201355, 53.7106934, -77.8960953, 49.5628510, -133.9829865, 131.6067810
3: -89.5787659, 46.3359642, -82.7529602, 42.8445358, -132.4233093, 129.0889130
4: -83.6508713, 62.1897888, -77.2107468, 57.3844490, -141.0353088, 139.4005432
5: -74.3674393, 57.6918259, -68.6543655, 53.2973862, -127.6648254, 126.3461914
6: -68.9480286, 67.5602417, -63.6260109, 62.3147240, -131.2627411, 131.1862488
7: -75.7947693, 67.0040512, -69.9391632, 61.8913269, -137.6860962, 136.9432068
8: -97.0751343, 58.4750214, -89.6712799, 53.9031601, -150.9782715, 148.1463013
9: -68.3939209, 68.3609619, -63.1125412, 63.0639992, -131.4579163, 131.4734955

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8913213, upper bound: 132.8875016
time: 10.96 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8848230, upper bound: 132.8790059
time: 9.07 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -70.5074997, 56.4462509, -70.6837006, 56.5846901, -127.0921936, 127.1299515
1: -60.7105560, 49.6791878, -60.8618927, 49.8032379, -110.5137939, 110.5410690
2: -78.5063171, 49.9475060, -78.7016602, 50.0718803, -128.5781555, 128.6491394
3: -83.4002838, 43.1676445, -83.6095352, 43.2728920, -126.6731720, 126.7771759
4: -77.8154297, 57.8254929, -78.0087433, 57.9687424, -135.7841797, 135.8341980
5: -69.1851654, 53.7049751, -69.3549957, 53.8359833, -123.0211487, 123.0599670
6: -64.1218719, 62.8021736, -64.2819290, 62.9590263, -127.0808868, 127.0841064
7: -70.4880142, 62.3697014, -70.6644669, 62.5212440, -133.0092621, 133.0341644
8: -90.3661499, 54.3129501, -90.5866013, 54.4484024, -144.8145294, 144.8995514
9: -63.6038513, 63.5527725, -63.7630730, 63.7096710, -127.3135071, 127.3158417

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9190470, upper bound: 132.9197241
time: 8.86 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9202565, upper bound: 132.9202565
time: 8.07 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 18.09 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 18.09
Output dim: 8, lower bound: -132.8913213, upper bound: 132.8875016
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 18.09
Output dim: 8, lower bound: -132.8848230, upper bound: 132.8790059
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 18.09
Output dim: 8, lower bound: -132.9190470, upper bound: 132.9197241
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 18.09
Output dim: 8, lower bound: -132.9202565, upper bound: 132.9202565

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -70.3093872, 56.2994804, -68.3067322, 54.6977234, -125.0071106, 124.6062164
1: -60.6548500, 49.4549751, -58.8482628, 48.1092224, -108.7640533, 108.3032379
2: -78.3835602, 49.7690086, -76.0754547, 48.3735352, -126.7570648, 125.8444672
3: -83.0765533, 42.8937683, -80.7882538, 41.8054543, -124.8820038, 123.6820221
4: -77.7121277, 57.6184082, -75.4206543, 56.0061760, -133.7183075, 133.0390625
5: -69.0565186, 53.5991478, -67.0521698, 52.0621948, -121.1187134, 120.6513214
6: -63.9218826, 62.6290054, -62.1086311, 60.8281631, -124.7500305, 124.7376404
7: -70.2517090, 62.3868942, -68.2667923, 60.4973488, -130.7490540, 130.6536865
8: -90.5281219, 53.8952713, -87.6973648, 52.5220604, -143.0501709, 141.5926361
9: -63.3239594, 63.3284836, -61.5819702, 61.5471458, -124.8711090, 124.9104538

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8843751, upper bound: 132.8789659
time: 8.43 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8843751, upper bound: 132.8790059
time: 9.73 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -74.5118408, 59.6168594, -67.4205322, 53.9927101, -128.5045471, 127.0373764
1: -64.3226776, 52.4077606, -58.0976181, 47.4790726, -111.8017502, 110.5053787
2: -83.1192322, 52.7136497, -75.0973663, 47.7367134, -130.8559418, 127.8110199
3: -88.1514053, 45.3751297, -79.7500153, 41.2572327, -129.4086304, 125.1251297
4: -82.4462814, 61.0329666, -74.4664001, 55.2728500, -137.7191315, 135.4993439
5: -73.1934814, 56.7817917, -66.1956482, 51.4025078, -124.5959854, 122.9774323
6: -67.7531433, 66.3885803, -61.2987595, 60.0348320, -127.7879791, 127.6873398
7: -74.5166626, 66.1696091, -67.3767853, 59.7513351, -134.2679901, 133.5463715
8: -96.0195999, 56.9180832, -86.6287384, 51.7920341, -147.8116302, 143.5467834
9: -67.1099930, 67.0907974, -60.7734833, 60.7408905, -127.8508759, 127.8642807

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8843751, upper bound: 132.8789659
time: 8.32 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8843751, upper bound: 132.8790059
time: 10.00 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -68.8505936, 55.1267891, -65.2420654, 52.2499275, -121.1005249, 120.3688507
1: -59.3183365, 48.4901657, -56.2837639, 45.9009018, -105.2192307, 104.7739258
2: -76.6832428, 48.7568359, -72.7158813, 46.1605721, -122.8438110, 121.4727097
3: -81.4331589, 42.1269875, -77.1514893, 39.8565636, -121.2897186, 119.2784729
4: -76.0233154, 56.4451218, -72.1226959, 53.4341507, -129.4574432, 128.5678101
5: -67.5810089, 52.4684181, -64.0869293, 49.7719688, -117.3529816, 116.5553360
6: -62.6021614, 61.3138542, -59.2973900, 58.0655937, -120.6677551, 120.6112442
7: -68.8133469, 60.9738884, -65.1676102, 57.9393730, -126.7527161, 126.1414948
8: -88.3894119, 52.9300346, -84.0978088, 49.9053078, -138.2947235, 137.0278473
9: -62.0713043, 62.0340843, -58.7354584, 58.7163086, -120.7876053, 120.7695389

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8843751, upper bound: 132.9189966
time: 10.88 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8843751, upper bound: 132.9197241
time: 9.96 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -67.9693222, 54.4256401, -69.5974808, 55.6752396, -123.6445618, 124.0231018
1: -58.5721436, 47.8633385, -60.0971375, 48.9616241, -107.5337677, 107.9604568
2: -75.7106934, 48.1231689, -77.6256561, 49.2082863, -124.9189758, 125.7488174
3: -80.4007797, 41.5818939, -82.4195938, 42.4158249, -122.8165894, 124.0014648
4: -75.0740204, 55.7159805, -77.0379486, 56.9671783, -132.0411987, 132.7539368
5: -66.7293472, 51.8123703, -68.3729553, 53.0688896, -119.7982330, 120.1853256
6: -61.7963867, 60.5251198, -63.2544403, 61.9612122, -123.7575836, 123.7795563
7: -67.9281464, 60.2321014, -69.5783997, 61.8703766, -129.7985229, 129.8105011
8: -87.3270569, 52.2033348, -89.8109360, 52.9954147, -140.3224640, 142.0142670
9: -61.2667160, 61.2325821, -62.6431046, 62.6076050, -123.8743210, 123.8756866

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9197241, upper bound: 132.9190356
time: 9.08 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9197241, upper bound: 132.9202565
time: 8.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.03 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 19.03
Output dim: 8, lower bound: -132.8843751, upper bound: 132.8789659
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 19.03
Output dim: 8, lower bound: -132.8843751, upper bound: 132.8790059
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 19.03
Output dim: 8, lower bound: -132.8843751, upper bound: 132.8789659
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 19.03
Output dim: 8, lower bound: -132.8843751, upper bound: 132.8790059
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.03
Output dim: 8, lower bound: -132.8843751, upper bound: 132.9189966
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.03
Output dim: 8, lower bound: -132.8843751, upper bound: 132.9197241
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.03
Output dim: 8, lower bound: -132.9197241, upper bound: 132.9190356
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.03
Output dim: 8, lower bound: -132.9197241, upper bound: 132.9202565

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -70.3093872, 56.2994804, -64.5335388, 51.6906662, -122.0000534, 120.8330231
1: -60.6548500, 49.4549751, -55.6719055, 45.4045525, -106.0593948, 105.1268768
2: -78.3835602, 49.7690086, -71.9256439, 45.6619034, -124.0454407, 121.6946564
3: -83.0765533, 42.8937683, -76.3103561, 39.4370270, -122.5135803, 119.2041245
4: -77.7121277, 57.6184082, -71.3393250, 52.8617249, -130.5738525, 128.9577026
5: -69.0565186, 53.5991478, -63.3992920, 49.2428055, -118.2993164, 116.9984360
6: -63.9218826, 62.6290054, -58.6551933, 57.4330292, -121.3548965, 121.2841949
7: -70.2517090, 62.3868942, -64.4564972, 57.3210068, -127.5727158, 126.8433914
8: -90.5281219, 53.8952713, -83.1994705, 49.3720703, -139.9001770, 137.0947418
9: -63.3239594, 63.3284836, -58.0983162, 58.0828934, -121.4068527, 121.4267960

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8677373, upper bound: 132.8643419
time: 14.48 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8655719, upper bound: 132.8608921
time: 11.88 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -70.3093872, 56.2994804, -68.8609009, 55.0946121, -125.4039993, 125.1603851
1: -60.6548500, 49.4549751, -59.4603500, 48.4451752, -109.1000214, 108.9153290
2: -78.3835602, 49.7690086, -76.8042221, 48.6897964, -127.0733337, 126.5732269
3: -83.0765533, 42.8937683, -81.5449219, 41.9789238, -125.0554657, 124.4386902
4: -77.7121277, 57.6184082, -76.2240677, 56.3716736, -134.0837860, 133.8424683
5: -69.0565186, 53.5991478, -67.6578674, 52.5189781, -121.5754852, 121.2570114
6: -63.9218826, 62.6290054, -62.5867844, 61.3036079, -125.2254639, 125.2157898
7: -70.2517090, 62.3868942, -68.8393402, 61.2280121, -131.4797211, 131.2262268
8: -90.5281219, 53.8952713, -88.8775864, 52.4408112, -142.9689178, 142.7728577
9: -63.3239594, 63.3284836, -61.9809608, 61.9486237, -125.2725830, 125.3094482

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8677373, upper bound: 132.8643419
time: 9.27 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8655719, upper bound: 132.8608921
time: 10.98 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -74.5118408, 59.6168594, -64.5335388, 51.6906662, -126.2025070, 124.1503983
1: -64.3226776, 52.4077606, -55.6719055, 45.4045525, -109.7272263, 108.0796661
2: -83.1192322, 52.7136497, -71.9256439, 45.6619034, -128.7811279, 124.6392899
3: -88.1514053, 45.3751297, -76.3103561, 39.4370270, -127.5884247, 121.6854706
4: -82.4462814, 61.0329666, -71.3393250, 52.8617249, -135.3080139, 132.3722534
5: -73.1934814, 56.7817917, -63.3992920, 49.2428055, -122.4362869, 120.1810684
6: -67.7531433, 66.3885803, -58.6551933, 57.4330292, -125.1861649, 125.0437775
7: -74.5166626, 66.1696091, -64.4564972, 57.3210068, -131.8376770, 130.6260681
8: -96.0195999, 56.9180832, -83.1994705, 49.3720703, -145.3916626, 140.1175537
9: -67.1099930, 67.0907974, -58.0983162, 58.0828934, -125.1928711, 125.1891174

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8605786, upper bound: 132.8562475
time: 8.60 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8567728, upper bound: 132.8504882
time: 7.47 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -74.5118408, 59.6168594, -68.8609009, 55.0946121, -129.6064453, 128.4777527
1: -64.3226776, 52.4077606, -59.4603500, 48.4451752, -112.7678528, 111.8681107
2: -83.1192322, 52.7136497, -76.8042221, 48.6897964, -131.8090210, 129.5178528
3: -88.1514053, 45.3751297, -81.5449219, 41.9789238, -130.1303253, 126.9200516
4: -82.4462814, 61.0329666, -76.2240677, 56.3716736, -138.8179321, 137.2570038
5: -73.1934814, 56.7817917, -67.6578674, 52.5189781, -125.7124481, 124.4396439
6: -67.7531433, 66.3885803, -62.5867844, 61.3036079, -129.0567322, 128.9753723
7: -74.5166626, 66.1696091, -68.8393402, 61.2280121, -135.7446747, 135.0089264
8: -96.0195999, 56.9180832, -88.8775864, 52.4408112, -148.4604187, 145.7956543
9: -67.1099930, 67.0907974, -61.9809608, 61.9486237, -129.0586090, 129.0717621

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8605786, upper bound: 132.8562475
time: 9.59 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8567728, upper bound: 132.8504882
time: 8.58 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -65.0691071, 52.1140366, -65.2420654, 52.2499275, -117.3190308, 117.3561020
1: -56.1351242, 45.7792931, -56.2837639, 45.9009018, -102.0360107, 102.0630569
2: -72.5241547, 46.0391197, -72.7158813, 46.1605721, -118.6847229, 118.7549973
3: -76.9457245, 39.7533722, -77.1514893, 39.8565636, -116.8022842, 116.9048615
4: -71.9329376, 53.2940903, -72.1226959, 53.4341507, -125.3670883, 125.4167786
5: -63.9201965, 49.6434708, -64.0869293, 49.7719688, -113.6921692, 113.7303848
6: -59.1408997, 57.9115448, -59.2973900, 58.0655937, -117.2064972, 117.2089386
7: -64.9947739, 57.7908134, -65.1676102, 57.9393730, -122.9341431, 122.9584198
8: -83.8812790, 49.7732430, -84.0978088, 49.9053078, -133.7865906, 133.8710480
9: -58.5801010, 58.5627518, -58.7354584, 58.7163086, -117.2964020, 117.2982025

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8846544, upper bound: 132.8891539
time: 10.05 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9190274, upper bound: 132.9190266
time: 8.43 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -69.4205017, 55.5362740, -65.2420654, 52.2499275, -121.6704254, 120.7783356
1: -59.9449081, 48.8372421, -56.2837639, 45.9009018, -105.8457870, 105.1210022
2: -77.4293976, 49.0839424, -72.7158813, 46.1605721, -123.5899658, 121.7998199
3: -82.2090836, 42.3102951, -77.1514893, 39.8565636, -122.0656357, 119.4617767
4: -76.8436890, 56.8238564, -72.1226959, 53.4341507, -130.2778320, 128.9465485
5: -68.2021790, 52.9374504, -64.0869293, 49.7719688, -117.9741516, 117.0243835
6: -63.0942955, 61.8035393, -59.2973900, 58.0655937, -121.1598816, 121.1009216
7: -69.4014893, 61.7182159, -65.1676102, 57.9393730, -127.3408661, 126.8858261
8: -89.5891571, 52.8604546, -84.0978088, 49.9053078, -139.4944611, 136.9582672
9: -62.4841614, 62.4503670, -58.7354584, 58.7163086, -121.2004471, 121.1858215

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8846544, upper bound: 132.8891539
time: 9.41 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9190274, upper bound: 132.9197241
time: 8.26 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -65.0691071, 52.1140366, -69.5974808, 55.6752396, -120.7443466, 121.7115021
1: -56.1351242, 45.7792931, -60.0971375, 48.9616241, -105.0967407, 105.8764343
2: -72.5241547, 46.0391197, -77.6256561, 49.2082863, -121.7324371, 123.6647415
3: -76.9457245, 39.7533722, -82.4195938, 42.4158249, -119.3615494, 122.1729660
4: -71.9329376, 53.2940903, -77.0379486, 56.9671783, -128.9001160, 130.3320312
5: -63.9201965, 49.6434708, -68.3729553, 53.0688896, -116.9890900, 118.0164185
6: -59.1408997, 57.9115448, -63.2544403, 61.9612122, -121.1021118, 121.1659851
7: -64.9947739, 57.7908134, -69.5783997, 61.8703766, -126.8651505, 127.3691864
8: -83.8812790, 49.7732430, -89.8109360, 52.9954147, -136.8766937, 139.5841827
9: -58.5801010, 58.5627518, -62.6431046, 62.6076050, -121.1877060, 121.2058563

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8827913, upper bound: 132.8878191
time: 9.50 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9189966, upper bound: 132.9190356
time: 7.69 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -69.4205017, 55.5362740, -69.5974808, 55.6752396, -125.0957336, 125.1337280
1: -59.9449081, 48.8372421, -60.0971375, 48.9616241, -108.9065170, 108.9343719
2: -77.4293976, 49.0839424, -77.6256561, 49.2082863, -126.6376801, 126.7095871
3: -82.2090836, 42.3102951, -82.4195938, 42.4158249, -124.6248779, 124.7298889
4: -76.8436890, 56.8238564, -77.0379486, 56.9671783, -133.8108673, 133.8618011
5: -68.2021790, 52.9374504, -68.3729553, 53.0688896, -121.2710571, 121.3104095
6: -63.0942955, 61.8035393, -63.2544403, 61.9612122, -125.0554810, 125.0579834
7: -69.4014893, 61.7182159, -69.5783997, 61.8703766, -131.2718658, 131.2966156
8: -89.5891571, 52.8604546, -89.8109360, 52.9954147, -142.5845642, 142.6713867
9: -62.4841614, 62.4503670, -62.6431046, 62.6076050, -125.0917511, 125.0934753

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8827913, upper bound: 132.8879508
time: 9.77 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9189966, upper bound: 132.9202565
time: 6.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 17.50 seconds
IS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 8, lower bound: -132.8677373, upper bound: 132.8643419
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 8, lower bound: -132.8655719, upper bound: 132.8608921
IS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 8, lower bound: -132.8677373, upper bound: 132.8643419
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 8, lower bound: -132.8655719, upper bound: 132.8608921
IS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 8, lower bound: -132.8605786, upper bound: 132.8562475
IS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 8, lower bound: -132.8567728, upper bound: 132.8504882
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 8, lower bound: -132.8605786, upper bound: 132.8562475
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 8, lower bound: -132.8567728, upper bound: 132.8504882
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 8, lower bound: -132.8846544, upper bound: 132.8891539
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 8, lower bound: -132.9190274, upper bound: 132.9190266
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 8, lower bound: -132.8846544, upper bound: 132.8891539
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 8, lower bound: -132.9190274, upper bound: 132.9197241
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 8, lower bound: -132.8827913, upper bound: 132.8878191
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 8, lower bound: -132.9189966, upper bound: 132.9190356
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 8, lower bound: -132.8827913, upper bound: 132.8879508
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.50
Output dim: 8, lower bound: -132.9189966, upper bound: 132.9202565

## BFS IS instance: IS_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -60.4500885, 48.3803444, -62.2766190, 49.8770485, -110.3271332, 110.6569595
1: -52.2669868, 42.4472656, -53.7548790, 43.7964668, -96.0634537, 96.2021484
2: -67.4159470, 42.7761040, -69.4139328, 44.0627327, -111.4786682, 112.1900330
3: -71.3466263, 36.8460350, -73.6181793, 38.0516510, -109.3982620, 110.4642181
4: -66.9204941, 49.4447021, -68.8623276, 50.9941597, -117.9146423, 118.3070221
5: -59.3458824, 46.1062317, -61.1778755, 47.5219345, -106.8678131, 107.2841034
6: -54.9207230, 53.7901344, -56.5941010, 55.4083328, -110.3290558, 110.3842239
7: -60.3241272, 53.8464127, -62.1811104, 55.3647995, -115.6889267, 116.0274963
8: -78.4259262, 45.9684906, -80.4231415, 47.5565300, -125.9824524, 126.3916321
9: -54.2160683, 54.3818626, -56.0123558, 56.0349083, -110.2509766, 110.3942184

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8691953, upper bound: 132.8643882
time: 11.64 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8691953, upper bound: 132.8645614
time: 10.23 seconds

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -63.6625404, 50.9070816, -58.7116928, 47.0230942, -110.6856308, 109.6187744
1: -55.1057892, 44.6695938, -50.7151260, 41.2679825, -96.3737717, 95.3847122
2: -71.0108490, 45.0019531, -65.4412994, 41.5390053, -112.5498505, 110.4432449
3: -75.2154541, 38.7104378, -69.3622055, 35.8745117, -111.0899658, 108.0726471
4: -70.5719223, 51.9959831, -64.9498596, 48.0416069, -118.6135254, 116.9458466
5: -62.5093307, 48.5708160, -57.6831436, 44.8367195, -107.3460388, 106.2539597
6: -57.8801613, 56.6258926, -53.3379250, 52.2117386, -110.0919037, 109.9638062
7: -63.5644188, 56.8004150, -58.5835037, 52.2823105, -115.8467255, 115.3839188
8: -82.7444077, 48.1467781, -76.0225372, 44.7199440, -127.4643402, 124.1693115
9: -57.0641212, 57.2136993, -52.7366562, 52.8169670, -109.8810883, 109.9503555

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8691953, upper bound: 132.8643882
time: 9.79 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8691953, upper bound: 132.8645614
time: 9.42 seconds

## BFS IS instance: IS_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -60.4500885, 48.3803444, -66.5707703, 53.2583618, -113.7084503, 114.9511108
1: -52.2669868, 42.4472656, -57.5144005, 46.8134079, -99.0803986, 99.9616699
2: -67.4159470, 42.7761040, -74.2588501, 47.0675278, -114.4834747, 117.0349579
3: -71.3466263, 36.8460350, -78.8173523, 40.5711899, -111.9178085, 115.6633835
4: -66.9204941, 49.4447021, -73.7159500, 54.4751053, -121.3955994, 123.1606522
5: -59.3458824, 46.1062317, -65.4067459, 50.7746925, -110.1205750, 111.5129776
6: -54.9207230, 53.7901344, -60.4992027, 59.2489471, -114.1696701, 114.2893372
7: -60.3241272, 53.8464127, -66.5341187, 59.2476540, -119.5717773, 120.3805161
8: -78.4259262, 45.9684906, -86.0680923, 50.5939827, -129.0199127, 132.0365906
9: -54.2160683, 54.3818626, -59.8657455, 59.8672333, -114.0832977, 114.2475967

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8646659, upper bound: 132.8607401
time: 9.40 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8646659, upper bound: 132.8608921
time: 9.81 seconds

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -63.6625404, 50.9070816, -63.0158386, 50.4181671, -114.0807037, 113.9229202
1: -55.1057892, 44.6695938, -54.4841232, 44.2928085, -99.3985977, 99.1537170
2: -71.0108490, 45.0019531, -70.2953491, 44.5543556, -115.5652008, 115.2972870
3: -75.2154541, 38.7104378, -74.5718918, 38.4092407, -113.6246796, 113.2823181
4: -70.5719223, 51.9959831, -69.8071747, 51.5407944, -122.1127167, 121.8031464
5: -62.5093307, 48.5708160, -61.9278030, 48.0951614, -110.6044846, 110.4986191
6: -57.8801613, 56.6258926, -57.2577629, 56.0630150, -113.9431686, 113.8836365
7: -63.5644188, 56.8004150, -62.9486237, 56.1690826, -119.7335052, 119.7490387
8: -82.7444077, 48.1467781, -81.6692734, 47.7783241, -130.5227203, 129.8160553
9: -57.0641212, 57.2136993, -56.6028061, 56.6600418, -113.7241669, 113.8164978

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8646659, upper bound: 132.8607401
time: 9.56 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8646659, upper bound: 132.8608921
time: 11.68 seconds

## BFS IS instance: IS_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -64.5559082, 51.6243935, -62.2766190, 49.8770485, -114.4329529, 113.9010086
1: -55.8643608, 45.3303299, -53.7548790, 43.7964668, -99.6608200, 99.0852051
2: -72.0532684, 45.6526756, -69.4139328, 44.0627327, -116.1159973, 115.0666046
3: -76.3089218, 39.2646751, -73.6181793, 38.0516510, -114.3605652, 112.8828278
4: -71.5528717, 52.7826996, -68.8623276, 50.9941597, -122.5470276, 121.6450195
5: -63.3988075, 49.2165070, -61.1778755, 47.5219345, -110.9207458, 110.3943787
6: -58.6667404, 57.4651642, -56.5941010, 55.4083328, -114.0750656, 114.0592575
7: -64.4952774, 57.5615158, -62.1811104, 55.3647995, -119.8600769, 119.7426300
8: -83.8169632, 48.8852615, -80.4231415, 47.5565300, -131.3734894, 129.3084106
9: -57.9100304, 58.0513687, -56.0123558, 56.0349083, -113.9449387, 114.0637207

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8559550, upper bound: 132.8503887
time: 14.26 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8559550, upper bound: 132.8505395
time: 11.12 seconds

## BFS IS instance: IS_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -67.7429581, 54.1312141, -58.7116928, 47.0230942, -114.7660522, 112.8429108
1: -58.6695862, 47.5286102, -50.7151260, 41.2679825, -99.9375610, 98.2437363
2: -75.6019287, 47.8568382, -65.4412994, 41.5390053, -117.1409302, 113.2981262
3: -80.1337357, 41.1164856, -69.3622055, 35.8745117, -116.0082474, 110.4786911
4: -75.1595993, 55.3124199, -64.9498596, 48.0416069, -123.2012024, 120.2622681
5: -66.5340805, 51.6535034, -57.6831436, 44.8367195, -111.3707962, 109.3366470
6: -61.5986557, 60.2695999, -53.3379250, 52.2117386, -113.8103943, 113.6075287
7: -67.6957245, 60.4671631, -58.5835037, 52.2823105, -119.9780350, 119.0506668
8: -88.0643692, 51.0771332, -76.0225372, 44.7199440, -132.7843018, 127.0996704
9: -60.7346954, 60.8544846, -52.7366562, 52.8169670, -113.5516510, 113.5911407

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8559550, upper bound: 132.8503887
time: 10.62 seconds

## Relational analysis of IS_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8559550, upper bound: 132.8505395
time: 12.15 seconds

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -64.5559082, 51.6243935, -66.5707703, 53.2583618, -117.8142700, 118.1951599
1: -55.8643608, 45.3303299, -57.5144005, 46.8134079, -102.6777573, 102.8447266
2: -72.0532684, 45.6526756, -74.2588501, 47.0675278, -119.1207886, 119.9115295
3: -76.3089218, 39.2646751, -78.8173523, 40.5711899, -116.8801117, 118.0820084
4: -71.5528717, 52.7826996, -73.7159500, 54.4751053, -126.0279770, 126.4986496
5: -63.3988075, 49.2165070, -65.4067459, 50.7746925, -114.1735001, 114.6232529
6: -58.6667404, 57.4651642, -60.4992027, 59.2489471, -117.9156876, 117.9643631
7: -64.4952774, 57.5615158, -66.5341187, 59.2476540, -123.7429199, 124.0956345
8: -83.8169632, 48.8852615, -86.0680923, 50.5939827, -134.4109497, 134.9533539
9: -57.9100304, 58.0513687, -59.8657455, 59.8672333, -117.7772675, 117.9171066

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8559182, upper bound: 132.8503240
time: 12.82 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8559182, upper bound: 132.8504882
time: 11.10 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -67.7429581, 54.1312141, -63.0158386, 50.4181671, -118.1611176, 117.1470490
1: -58.6695862, 47.5286102, -54.4841232, 44.2928085, -102.9623795, 102.0127335
2: -75.6019287, 47.8568382, -70.2953491, 44.5543556, -120.1562729, 118.1521759
3: -80.1337357, 41.1164856, -74.5718918, 38.4092407, -118.5429688, 115.6883698
4: -75.1595993, 55.3124199, -69.8071747, 51.5407944, -126.7003937, 125.1195831
5: -66.5340805, 51.6535034, -61.9278030, 48.0951614, -114.6292419, 113.5813065
6: -61.5986557, 60.2695999, -57.2577629, 56.0630150, -117.6616669, 117.5273590
7: -67.6957245, 60.4671631, -62.9486237, 56.1690826, -123.8648071, 123.4157867
8: -88.0643692, 51.0771332, -81.6692734, 47.7783241, -135.8426819, 132.7463989
9: -60.7346954, 60.8544846, -56.6028061, 56.6600418, -117.3947296, 117.4572906

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8559182, upper bound: 132.8503240
time: 12.17 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8559182, upper bound: 132.8504882
time: 9.83 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -64.6670380, 51.7952881, -73.1926727, 58.5592575, -123.2262878, 124.9879608
1: -55.7864761, 45.4981689, -63.0287552, 51.4743538, -107.2608337, 108.5269241
2: -72.0739746, 45.7580872, -81.5190659, 51.7994537, -123.8734131, 127.2771530
3: -76.4650650, 39.5136108, -86.4288635, 44.6686897, -121.1337433, 125.9424744
4: -71.4844131, 52.9657745, -80.6817017, 59.8828354, -131.3672485, 133.6474762
5: -63.5268288, 49.3409538, -71.7688599, 55.7079697, -119.2347946, 121.1098175
6: -58.7743416, 57.5522041, -66.5323105, 65.1386261, -123.9129562, 124.0845032
7: -64.5897446, 57.4372559, -73.0872726, 64.7306061, -129.3203430, 130.5245209
8: -83.3657761, 49.4692001, -93.8213196, 56.2135124, -139.5792847, 143.2905121
9: -58.2170410, 58.2004204, -65.8614807, 65.7923508, -124.0093689, 124.0619049

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8530764, upper bound: 132.8558005
time: 10.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8395243, upper bound: 132.8389600
time: 10.27 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8693985, upper bound: 132.8750754
time: 9.13 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8704709, upper bound: 132.8759930
time: 8.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -65.0691071, 52.1140366, -65.1598511, 52.1848068, -117.2539139, 117.2738876
1: -56.1351242, 45.7792931, -56.2123833, 45.8433456, -101.9784622, 101.9916763
2: -72.5241547, 46.0391197, -72.6239929, 46.1032333, -118.6273804, 118.6631012
3: -76.9457245, 39.7533722, -77.0532684, 39.8072243, -116.7529449, 116.8066406
4: -71.9329376, 53.2940903, -72.0311279, 53.3666992, -125.2996368, 125.3252106
5: -63.9201965, 49.6434708, -64.0065002, 49.7100372, -113.6302338, 113.6499557
6: -59.1408997, 57.9115448, -59.2224350, 57.9920845, -117.1329803, 117.1339798
7: -64.9947739, 57.7908134, -65.0848770, 57.8674316, -122.8621826, 122.8756866
8: -83.8812790, 49.7732430, -83.9928055, 49.8425941, -133.7238617, 133.7660522
9: -58.5801010, 58.5627518, -58.6611748, 58.6421204, -117.2222137, 117.2239151

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9002742, upper bound: 132.8998345
time: 9.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9018493, upper bound: 132.9018493
time: 6.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -69.0108414, 55.2117882, -73.1926727, 58.5592575, -127.5700989, 128.4044647
1: -59.5896263, 48.5506630, -63.0287552, 51.4743538, -111.0639801, 111.5793991
2: -76.9707413, 48.7973709, -81.5190659, 51.7994537, -128.7702026, 130.3164215
3: -81.7196732, 42.0660019, -86.4288635, 44.6686897, -126.3883514, 128.4948578
4: -76.3872528, 56.4892387, -80.6817017, 59.8828354, -136.2700653, 137.1709137
5: -67.8013535, 52.6296005, -71.7688599, 55.7079697, -123.5093155, 124.3984604
6: -62.7209930, 61.4373131, -66.5323105, 65.1386261, -127.8596115, 127.9696121
7: -68.9889297, 61.3583984, -73.0872726, 64.7306061, -133.7195129, 134.4456787
8: -89.0645905, 52.5502586, -93.8213196, 56.2135124, -145.2781067, 146.3715210
9: -62.1144104, 62.0811920, -65.8614807, 65.7923508, -127.9067078, 127.9426575

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8518544, upper bound: 132.8542184
time: 10.98 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8285830, upper bound: 132.8280698
time: 11.05 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=145.0349884033203
rel_dist={8: [-132.92639238641755, 132.92639237568028]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9099773, upper bound: 132.9089337
time: 9.71 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9170978, upper bound: 132.9170978
time: 9.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.68 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.68
Output dim: 8, lower bound: -132.9099773, upper bound: 132.9089337
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.68
Output dim: 8, lower bound: -132.9170978, upper bound: 132.9170978

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -56.5003929, 45.1903267, -59.6667824, 47.7414589, -104.2418518, 104.8571091
1: -48.8756294, 39.6732941, -51.5521393, 41.9355698, -90.8111877, 91.2254181
2: -62.9268761, 39.9484482, -66.4486694, 42.2103233, -105.1371918, 106.3971176
3: -66.6489029, 34.4047318, -70.4367981, 36.3942719, -103.0431747, 104.8415298
4: -62.7137680, 46.2145767, -66.1279984, 48.8487663, -111.5625305, 112.3425751
5: -55.5814896, 43.1901779, -58.6674347, 45.5671501, -101.1486130, 101.8575974
6: -51.2055817, 50.2586021, -54.1324654, 53.0927963, -104.2983780, 104.3910675
7: -56.3788567, 50.4917526, -59.5711327, 53.1771584, -109.5560150, 110.0628815
8: -73.3426361, 42.7920685, -77.1899261, 45.3987656, -118.7414017, 119.9819946
9: -50.8569565, 50.9877281, -53.7411880, 53.8261223, -104.6830750, 104.7289124

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8886384, upper bound: 132.8879733
time: 8.97 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8913459, upper bound: 132.8902743
time: 9.38 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -65.4089584, 52.3559074, -65.0044861, 52.0528793, -117.4618225, 117.3603973
1: -56.4612579, 46.0189743, -56.0574570, 45.7598495, -102.2211075, 102.0764008
2: -72.8729858, 46.2825661, -72.3914566, 46.0254517, -118.8984375, 118.6740265
3: -77.3459702, 39.9459114, -76.8500595, 39.7568588, -117.1028290, 116.7959747
4: -72.3922043, 53.6026535, -71.8746109, 53.2942085, -125.6864166, 125.4772491
5: -64.2686691, 49.9036140, -63.8517876, 49.5872421, -113.8559113, 113.7554016
6: -59.4342346, 58.2340508, -59.0712700, 57.8782730, -117.3125076, 117.3053055
7: -65.3617859, 58.1372910, -64.9567108, 57.7000504, -123.0618362, 123.0940018
8: -84.3325882, 49.9750977, -83.6765747, 49.8270187, -134.1595917, 133.6516724
9: -58.9498329, 58.9564171, -58.6085167, 58.6106300, -117.5604630, 117.5649261

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8965116, upper bound: 132.8971263
time: 9.59 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8999956, upper bound: 132.8999956
time: 8.70 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.43 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 19.43
Output dim: 8, lower bound: -132.8886384, upper bound: 132.8879733
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 19.43
Output dim: 8, lower bound: -132.8913459, upper bound: 132.8902743
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 19.43
Output dim: 8, lower bound: -132.8965116, upper bound: 132.8971263
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 19.43
Output dim: 8, lower bound: -132.8999956, upper bound: 132.8999956

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -50.1644745, 40.1053238, -50.4291496, 40.3172607, -90.4817352, 90.5344620
1: -43.4821091, 35.1695557, -43.6899109, 35.3690109, -78.8511200, 78.8594513
2: -55.8729248, 35.4864807, -56.1694260, 35.6857376, -91.5586624, 91.6559067
3: -59.0971756, 30.4867458, -59.4149017, 30.6950493, -89.7922211, 89.9016495
4: -55.7517471, 40.9511375, -55.9914474, 41.1719780, -96.9237213, 96.9425812
5: -49.3373947, 38.3676033, -49.5558815, 38.5411148, -87.8784943, 87.9234848
6: -45.4304848, 44.5829315, -45.7002869, 44.8182220, -90.2487030, 90.2832108
7: -50.0010300, 44.9931564, -50.2615128, 45.1682739, -95.1692810, 95.2546692
8: -65.5267029, 37.7327271, -65.8115997, 38.0069427, -103.5336456, 103.5443192
9: -45.0077515, 45.2437439, -45.2043190, 45.4546700, -90.4624176, 90.4480515

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8884554, upper bound: 132.8874226
time: 8.81 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8884554, upper bound: 132.8879733
time: 8.55 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -47.5825462, 38.0501328, -53.6491508, 42.8419266, -90.4244690, 91.6992798
1: -41.2820740, 33.3487587, -46.5591927, 37.6021271, -78.8842010, 79.9079514
2: -52.9847984, 33.6675568, -59.7827225, 37.9134293, -90.8982239, 93.4502716
3: -56.0139313, 28.9073811, -63.3032990, 32.5554924, -88.5694199, 92.2106781
4: -52.9063263, 38.8110809, -59.6642418, 43.7371826, -96.6434937, 98.4753113
5: -46.8308601, 36.4391327, -52.7248650, 41.0125389, -87.8433914, 89.1640015
6: -43.0816002, 42.2740440, -48.6549339, 47.6716347, -90.7532349, 90.9289703
7: -47.3951225, 42.7602577, -53.5058441, 48.1408234, -95.5359344, 96.2660980
8: -62.2913742, 35.6997375, -70.1690674, 40.1724396, -102.4637985, 105.8688049
9: -42.6626472, 42.9100227, -48.0569458, 48.2935333, -90.9561691, 90.9669495

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8766589, upper bound: 132.8749907
time: 11.05 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8792274, upper bound: 132.8779618
time: 10.68 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -58.5275803, 46.8177834, -55.2525826, 44.2045059, -102.7320862, 102.0703659
1: -50.6085968, 41.1175156, -47.7623062, 38.8154602, -89.4240570, 88.8798141
2: -65.2177811, 41.4086075, -61.5435562, 39.1187744, -104.3365555, 102.9521637
3: -69.1231232, 35.7102661, -65.1996536, 33.7541885, -102.8773117, 100.9099197
4: -64.8451843, 47.8936501, -61.1761932, 45.1985207, -110.0437012, 109.0698395
5: -57.4777031, 44.6664543, -54.2307587, 42.1606293, -99.6383209, 98.8971939
6: -53.1482925, 52.0640373, -50.1627235, 49.1361580, -102.2844391, 102.2267609
7: -58.4177818, 52.1719170, -55.1145477, 49.2457237, -107.6634827, 107.2864532
8: -75.8760223, 44.4530792, -71.6825562, 42.0023193, -117.8783417, 116.1356354
9: -52.5833282, 52.7215385, -49.5869789, 49.7737885, -102.3571167, 102.3085175

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8962525, upper bound: 132.8962525
time: 9.51 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8962525, upper bound: 132.8971263
time: 8.73 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -55.7986145, 44.6476212, -58.6475487, 46.8742409, -102.6728516, 103.2951508
1: -48.2785378, 39.1920815, -50.7799950, 41.1731224, -89.4516525, 89.9720764
2: -62.1657982, 39.4757195, -65.3437881, 41.4712563, -103.6370392, 104.8195038
3: -65.8652573, 34.0689392, -69.2952423, 35.7338295, -101.5990906, 103.3641815
4: -61.8382645, 45.6386299, -65.0452728, 47.9061012, -109.7443695, 110.6838913
5: -54.8298988, 42.6297226, -57.5719872, 44.7666969, -99.5965881, 100.2017059
6: -50.6565819, 49.6204109, -53.2803268, 52.1426888, -102.7992706, 102.9007416
7: -55.6665421, 49.8161316, -58.5352173, 52.3634949, -108.0300369, 108.3513489
8: -72.4629059, 42.3146210, -76.2449265, 44.3180504, -116.7809601, 118.5595474
9: -50.0974350, 50.2652740, -52.5973587, 52.7705345, -102.8679657, 102.8626328

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8971263, upper bound: 132.8963251
time: 9.84 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8892271, upper bound: 132.8999956
time: 10.40 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.38 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.38
Output dim: 8, lower bound: -132.8884554, upper bound: 132.8874226
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.38
Output dim: 8, lower bound: -132.8884554, upper bound: 132.8879733
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.38
Output dim: 8, lower bound: -132.8766589, upper bound: 132.8749907
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.38
Output dim: 8, lower bound: -132.8792274, upper bound: 132.8779618
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.38
Output dim: 8, lower bound: -132.8962525, upper bound: 132.8962525
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.38
Output dim: 8, lower bound: -132.8962525, upper bound: 132.8971263
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.38
Output dim: 8, lower bound: -132.8971263, upper bound: 132.8963251
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.38
Output dim: 8, lower bound: -132.8892271, upper bound: 132.8999956

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -47.6909790, 38.1188507, -50.4291496, 40.3172607, -88.0082397, 88.5479813
1: -41.3771362, 33.4167442, -43.6899109, 35.3690109, -76.7461472, 77.1066589
2: -53.1178665, 33.7476768, -56.1694260, 35.6857376, -88.8035889, 89.9170990
3: -56.1451645, 28.9522667, -59.4149017, 30.6950493, -86.8402023, 88.3671722
4: -53.0300484, 38.8894463, -55.9914474, 41.1719780, -94.2020264, 94.8808899
5: -46.8992844, 36.4836884, -49.5558815, 38.5411148, -85.4403992, 86.0395660
6: -43.1797256, 42.3724518, -45.7002869, 44.8182220, -87.9979401, 88.0727386
7: -47.5053177, 42.8433762, -50.2615128, 45.1682739, -92.6735840, 93.1048889
8: -62.4663582, 35.7517776, -65.8115997, 38.0069427, -100.4732971, 101.5633774
9: -42.7237396, 42.9935837, -45.2043190, 45.4546700, -88.1784058, 88.1978912

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8729887, upper bound: 132.8723616
time: 7.54 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8759670, upper bound: 132.8748215
time: 8.68 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -50.9091072, 40.6380882, -50.4291496, 40.3172607, -91.2263641, 91.0672302
1: -44.2325211, 35.6431274, -43.6899109, 35.3690109, -79.6015320, 79.3330383
2: -56.7236824, 35.9708633, -56.1694260, 35.6857376, -92.4094162, 92.1402817
3: -60.0267029, 30.8067131, -59.4149017, 30.6950493, -90.7217407, 90.2216187
4: -56.6926384, 41.4501915, -55.9914474, 41.1719780, -97.8646164, 97.4416351
5: -50.0557938, 38.9531479, -49.5558815, 38.5411148, -88.5969086, 88.5090256
6: -46.1237068, 45.2162437, -45.7002869, 44.8182220, -90.9419250, 90.9165268
7: -50.7477646, 45.8103676, -50.2615128, 45.1682739, -95.9160309, 96.0718765
8: -66.8157349, 37.9088058, -65.8115997, 38.0069427, -104.8226776, 103.7203979
9: -45.5713730, 45.8221397, -45.2043190, 45.4546700, -91.0260468, 91.0264435

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8729887, upper bound: 132.8729040
time: 12.39 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8759670, upper bound: 132.8753681
time: 9.75 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -37.6760864, 30.0883121, -45.9542618, 36.6611481, -74.3372345, 76.0425568
1: -32.8401604, 26.3830338, -39.9996147, 32.1635513, -65.0037079, 66.3826447
2: -41.9790649, 26.6979313, -51.2278595, 32.4923096, -74.4713745, 77.9257889
3: -44.2485847, 22.7346153, -54.1532898, 27.7449875, -71.9935608, 76.8878937
4: -42.0361481, 30.5578270, -51.2153511, 37.3307190, -79.3668671, 81.7731781
5: -37.0778389, 28.9375095, -45.1518364, 35.1862602, -72.2640991, 74.0893326
6: -34.1050606, 33.4267998, -41.6635094, 40.7831192, -74.8881760, 75.0903015
7: -37.4509048, 34.1751251, -45.7848740, 41.4777527, -78.9286575, 79.9599915
8: -50.0221786, 27.7727623, -60.6598701, 33.9870567, -84.0092239, 88.4326248
9: -33.5614777, 33.9200783, -40.9745483, 41.3039246, -74.8654022, 74.8946228

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8508502, upper bound: 132.8481728
time: 9.68 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8500128, upper bound: 132.8478292
time: 10.77 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -41.4274940, 33.1216583, -47.3067207, 37.7625046, -79.1899872, 80.4283752
1: -36.0716591, 29.0389538, -41.1399994, 33.1273575, -69.1990128, 70.1789474
2: -46.1739044, 29.3213768, -52.7264175, 33.4368095, -79.6106949, 82.0477905
3: -48.7679329, 25.0734596, -55.7764130, 28.6266518, -77.3945847, 80.8498688
4: -46.2002716, 33.6847534, -52.7027893, 38.4776611, -84.6779327, 86.3875427
5: -40.8078461, 31.8130283, -46.5109978, 36.2309952, -77.0388336, 78.3240128
6: -37.5138702, 36.7829971, -42.8864861, 41.9923973, -79.5062714, 79.6694794
7: -41.2254410, 37.5002289, -47.1386108, 42.6587601, -83.8841934, 84.6388397
8: -54.7668076, 30.7214470, -62.2933388, 35.1038513, -89.8706589, 93.0147858
9: -37.0350151, 37.3127441, -42.2601929, 42.5325890, -79.5675888, 79.5729370

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8468628, upper bound: 132.8481728
time: 11.32 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8531979, upper bound: 132.8517781
time: 8.58 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -55.6988449, 44.5405273, -55.2525826, 44.2045059, -99.9033508, 99.7931061
1: -48.2030106, 39.1063004, -47.7623062, 38.8154602, -87.0184708, 86.8685989
2: -62.0717392, 39.4054909, -61.5435562, 39.1187744, -101.1905136, 100.9490356
3: -65.7460098, 33.9678421, -65.1996536, 33.7541885, -99.5001984, 99.1674957
4: -61.7432289, 45.5412979, -61.1761932, 45.1985207, -106.9417496, 106.7174835
5: -54.6878510, 42.5104713, -54.2307587, 42.1606293, -96.8484802, 96.7412262
6: -50.5640373, 49.5333138, -50.1627235, 49.1361580, -99.7001953, 99.6960297
7: -55.5634270, 49.7225037, -55.1145477, 49.2457237, -104.8091354, 104.8370514
8: -72.3952103, 42.1800880, -71.6825562, 42.0023193, -114.3975296, 113.8626404
9: -49.9690018, 50.1585464, -49.5869789, 49.7737885, -99.7427902, 99.7455215

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8778767, upper bound: 132.8776704
time: 10.76 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8791865, upper bound: 132.8790054
time: 9.13 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -59.0268059, 47.1576653, -55.2525826, 44.2045059, -103.2313080, 102.4102478
1: -51.1610718, 41.4182701, -47.7623062, 38.8154602, -89.9765320, 89.1805725
2: -65.7993317, 41.7126884, -61.5435562, 39.1187744, -104.9181061, 103.2562408
3: -69.7575684, 35.9106216, -65.1996536, 33.7541885, -103.5117569, 101.1102676
4: -65.5321121, 48.1966133, -61.1761932, 45.1985207, -110.7306366, 109.3728027
5: -57.9627724, 45.0639763, -54.2307587, 42.1606293, -100.1233978, 99.2947388
6: -53.6182785, 52.4826126, -50.1627235, 49.1361580, -102.7544250, 102.6453400
7: -58.9167023, 52.7781219, -55.1145477, 49.2457237, -108.1624222, 107.8926697
8: -76.8683319, 44.4505272, -71.6825562, 42.0023193, -118.8706512, 116.1330872
9: -52.9197235, 53.0957184, -49.5869789, 49.7737885, -102.6935120, 102.6826935

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8778767, upper bound: 132.8781133
time: 13.76 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8791865, upper bound: 132.8795759
time: 9.15 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -55.6927910, 44.5353622, -58.6475487, 46.8742409, -102.5670319, 103.1828995
1: -48.1970596, 39.1016960, -50.7799950, 41.1731224, -89.3701706, 89.8816833
2: -62.0639343, 39.4005737, -65.3437881, 41.4712563, -103.5351868, 104.7443619
3: -65.7369843, 33.9645309, -69.2952423, 35.7338295, -101.4708099, 103.2597733
4: -61.7353592, 45.5360336, -65.0452728, 47.9061012, -109.6414642, 110.5812912
5: -54.6822891, 42.5054703, -57.5719872, 44.7666969, -99.4489670, 100.0774460
6: -50.5583572, 49.5271873, -53.2803268, 52.1426888, -102.7010345, 102.8075104
7: -55.5563393, 49.7171936, -58.5352173, 52.3634949, -107.9198303, 108.2524109
8: -72.3869400, 42.1754379, -76.2449265, 44.3180504, -116.7049866, 118.4203644
9: -49.9630737, 50.1531372, -52.5973587, 52.7705345, -102.7336121, 102.7504959

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8775353, upper bound: 132.8775768
time: 8.44 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8788689, upper bound: 132.8789461
time: 8.91 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -59.0097885, 47.1433296, -58.6475487, 46.8742409, -105.8840256, 105.7908707
1: -51.1444092, 41.4053345, -50.7799950, 41.1731224, -92.3175049, 92.1853333
2: -65.7771912, 41.6981773, -65.3437881, 41.4712563, -107.2484436, 107.0419617
3: -69.7296829, 35.9015503, -69.2952423, 35.7338295, -105.4635010, 105.1967926
4: -65.5095520, 48.1816788, -65.0452728, 47.9061012, -113.4156494, 113.2269516
5: -57.9485550, 45.0508461, -57.5719872, 44.7666969, -102.7152557, 102.6228333
6: -53.6021080, 52.4646873, -53.2803268, 52.1426888, -105.7447968, 105.7450104
7: -58.8953629, 52.7652702, -58.5352173, 52.3634949, -111.2588577, 111.3004913
8: -76.8466263, 44.4354362, -76.2449265, 44.3180504, -121.1646729, 120.6803589
9: -52.9019966, 53.0811310, -52.5973587, 52.7705345, -105.6725159, 105.6784744

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8874226, upper bound: 132.8913459
time: 8.87 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8874226, upper bound: 132.8999956
time: 9.18 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 19.19 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.19
Output dim: 8, lower bound: -132.8729887, upper bound: 132.8723616
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.19
Output dim: 8, lower bound: -132.8759670, upper bound: 132.8748215
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.19
Output dim: 8, lower bound: -132.8729887, upper bound: 132.8729040
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.19
Output dim: 8, lower bound: -132.8759670, upper bound: 132.8753681
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.19
Output dim: 8, lower bound: -132.8508502, upper bound: 132.8481728
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.19
Output dim: 8, lower bound: -132.8500128, upper bound: 132.8478292
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.19
Output dim: 8, lower bound: -132.8468628, upper bound: 132.8481728
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.19
Output dim: 8, lower bound: -132.8531979, upper bound: 132.8517781
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.19
Output dim: 8, lower bound: -132.8778767, upper bound: 132.8776704
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.19
Output dim: 8, lower bound: -132.8791865, upper bound: 132.8790054
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.19
Output dim: 8, lower bound: -132.8778767, upper bound: 132.8781133
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.19
Output dim: 8, lower bound: -132.8791865, upper bound: 132.8795759
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 19.19
Output dim: 8, lower bound: -132.8775353, upper bound: 132.8775768
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 19.19
Output dim: 8, lower bound: -132.8788689, upper bound: 132.8789461
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.19
Output dim: 8, lower bound: -132.8874226, upper bound: 132.8913459
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.19
Output dim: 8, lower bound: -132.8874226, upper bound: 132.8999956

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -40.4323387, 32.2828522, -40.4799690, 32.3210907, -72.7534332, 72.7628174
1: -35.1938019, 28.3050003, -35.2124405, 28.3479576, -63.5417519, 63.5174408
2: -45.0505905, 28.6341038, -45.1068649, 28.6746941, -73.7252808, 73.7409668
3: -47.5194969, 24.4184456, -47.5829773, 24.4814415, -72.0009384, 72.0014191
4: -45.0593605, 32.8426819, -45.0708389, 32.8889122, -77.9482651, 77.9135208
5: -39.7578049, 30.9874401, -39.7791252, 31.0073013, -70.7650986, 70.7665558
6: -36.5942116, 35.8868141, -36.6722183, 35.9215355, -72.5157471, 72.5590210
7: -40.2171707, 36.5564270, -40.2710571, 36.5519257, -76.7690964, 76.8274841
8: -53.4766083, 29.9398441, -53.4972229, 30.0264854, -83.5030823, 83.4370575
9: -36.0568962, 36.4014359, -36.0569534, 36.4212227, -72.4781189, 72.4583893

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8762406, upper bound: 132.8756992
time: 8.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8762352, upper bound: 132.8756855
time: 9.26 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -41.2867661, 32.9911957, -43.7108841, 34.9347153, -76.2214813, 76.7020798
1: -35.9237061, 28.9218369, -37.9982491, 30.6404343, -66.5641327, 66.9200897
2: -46.0094872, 29.2265987, -48.7255173, 30.9373302, -76.9468155, 77.9520874
3: -48.5622711, 24.9824009, -51.4792671, 26.5135918, -75.0758667, 76.4616699
4: -46.0071259, 33.5727272, -48.6659317, 35.5725288, -81.5796356, 82.2386627
5: -40.6282501, 31.6593494, -42.9849777, 33.4861298, -74.1143799, 74.6443024
6: -37.3672676, 36.6532974, -39.6160851, 38.8144493, -76.1817093, 76.2693787
7: -41.0712776, 37.3228607, -43.5238190, 39.4224701, -80.4937439, 80.8466797
8: -54.5221100, 30.6414471, -57.5903702, 32.5621681, -87.0842667, 88.2318192
9: -36.8840714, 37.1807594, -39.0473824, 39.3396111, -76.2236786, 76.2281418

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8489408, upper bound: 132.8499579
time: 10.24 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8510328, upper bound: 132.8495100
time: 9.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -43.4367905, 34.6387444, -40.4799690, 32.3210907, -75.7578812, 75.1187134
1: -37.8750839, 30.3858318, -35.2124405, 28.3479576, -66.2230377, 65.5982666
2: -48.4248009, 30.7151241, -45.1068649, 28.6746941, -77.0994873, 75.8219910
3: -51.1589737, 26.1445103, -47.5829773, 24.4814415, -75.6404114, 73.7274857
4: -48.4953537, 35.2281761, -45.0708389, 32.8889122, -81.3842545, 80.2990036
5: -42.7009773, 33.2980499, -39.7791252, 31.0073013, -73.7082748, 73.0771713
6: -39.3443069, 38.5432930, -36.6722183, 35.9215355, -75.2658386, 75.2155075
7: -43.2531281, 39.3425140, -40.2710571, 36.5519257, -79.8050537, 79.6135712
8: -57.5790787, 31.9211617, -53.4972229, 30.0264854, -87.6055603, 85.4183807
9: -38.7076225, 39.0478210, -36.0569534, 36.4212227, -75.1288452, 75.1047745

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8461862, upper bound: 132.8459261
time: 11.29 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8451257, upper bound: 132.8455237
time: 9.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -44.7254448, 35.6921654, -43.7108841, 34.9347153, -79.6601562, 79.4030457
1: -38.9646835, 31.3037682, -37.9982491, 30.6404343, -69.6051178, 69.3020172
2: -49.8553619, 31.6140461, -48.7255173, 30.9373302, -80.7926941, 80.3395615
3: -52.7096176, 26.9795227, -51.4792671, 26.5135918, -79.2232056, 78.4587860
4: -49.9145355, 36.3238525, -48.6659317, 35.5725288, -85.4870529, 84.9897766
5: -44.0017128, 34.2948494, -42.9849777, 33.4861298, -77.4878387, 77.2798157
6: -40.5091133, 39.6961060, -39.6160851, 38.8144493, -79.3235474, 79.3121796
7: -44.5436897, 40.4722290, -43.5238190, 39.4224701, -83.9661407, 83.9960480
8: -59.1402550, 32.9866905, -57.5903702, 32.5621681, -91.7024231, 90.5770493
9: -39.9355965, 40.2205200, -39.0473824, 39.3396111, -79.2751923, 79.2678909

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8759670, upper bound: 132.8753608
time: 11.36 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8759670, upper bound: 132.8753608
time: 10.48 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -33.4252090, 26.7067299, -39.7733345, 31.7443504, -65.1695557, 66.4800644
1: -29.1855354, 23.4333267, -34.7124748, 27.8655205, -57.0510445, 58.1458015
2: -37.2649536, 23.6621246, -44.3649826, 28.0855160, -65.3504715, 68.0270996
3: -39.3140526, 20.1785069, -46.9676857, 24.0126953, -63.3267365, 67.1461868
4: -37.4290276, 27.0667534, -44.4911880, 32.2390900, -69.6681213, 71.5579300
5: -32.9642982, 25.7457924, -39.1650543, 30.5584335, -63.5227318, 64.9108429
6: -30.2330246, 29.6778145, -36.0219879, 35.3166084, -65.5496292, 65.6997910
7: -33.1791687, 30.5430450, -39.5755310, 36.2056694, -69.3848267, 70.1185760
8: -44.7589951, 24.3297462, -53.0274048, 28.9799404, -73.7389374, 77.3571472
9: -29.7384186, 30.1187191, -35.4221115, 35.7838974, -65.5223083, 65.5408325

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8508502, upper bound: 132.8481728
time: 10.47 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8508502, upper bound: 132.8481728
time: 9.14 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -30.7604904, 24.5674095, -37.7245026, 30.0997353, -60.8602257, 62.2919083
1: -26.9071541, 21.5883636, -32.9843864, 26.4546890, -53.3618317, 54.5727501
2: -34.3230286, 21.7712822, -42.1381760, 26.6088638, -60.9318810, 63.9094582
3: -36.2113342, 18.5753441, -44.6369247, 22.7753983, -58.9867249, 63.2122688
4: -34.5527267, 24.8820629, -42.3116341, 30.5542507, -65.1069794, 67.1936951
5: -30.3814716, 23.7576180, -37.2117996, 29.0454559, -59.4269257, 60.9694138
6: -27.8082619, 27.3247337, -34.1525192, 33.5187874, -61.3270493, 61.4772491
7: -30.5193729, 28.2877102, -37.5207672, 34.5403366, -65.0597076, 65.8084793
8: -41.4922638, 22.1380901, -50.6192474, 27.1752167, -68.6674805, 72.7573395
9: -27.3359356, 27.7079849, -33.5635223, 33.9214134, -61.2573395, 61.2715073

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8500128, upper bound: 132.8478292
time: 9.32 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8500128, upper bound: 132.8478292
time: 9.89 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -36.9053383, 29.5312347, -40.9363480, 32.7047577, -69.6100922, 70.4675674
1: -32.2047386, 25.9026947, -35.7050476, 28.7023697, -60.9071083, 61.6077423
2: -41.1645203, 26.1059570, -45.6620750, 28.9031010, -70.0676193, 71.7680359
3: -43.5236282, 22.3432770, -48.3815231, 24.7698059, -68.2934341, 70.7248001
4: -41.2974281, 29.9669342, -45.7738953, 33.2312469, -74.5286713, 75.7408142
5: -36.4286423, 28.4264870, -40.3430176, 31.4670563, -67.8956985, 68.7694855
6: -33.3978539, 32.7902794, -37.0760651, 36.3572655, -69.7551117, 69.8663483
7: -36.6944199, 33.6520920, -40.7483253, 37.2400551, -73.9344788, 74.4004211
8: -49.1910133, 27.0572357, -54.4517670, 29.9372063, -79.1282196, 81.5090027
9: -32.9774170, 33.2717628, -36.5424194, 36.8427086, -69.8201294, 69.8141785

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8542066, upper bound: 132.8521347
time: 10.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8542066, upper bound: 132.8521349
time: 8.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -34.2707443, 27.4308777, -39.0614014, 31.2034969, -65.4742432, 66.4922791
1: -29.9453964, 24.0801392, -34.1196594, 27.4113693, -57.3567619, 58.1997948
2: -38.2565460, 24.2336197, -43.6259689, 27.5479031, -65.8044510, 67.8595886
3: -40.4657097, 20.7590027, -46.2498932, 23.6350250, -64.1007156, 67.0088730
4: -38.4552765, 27.8035641, -43.7791710, 31.6894016, -70.1446762, 71.5827332
5: -33.8868904, 26.4555397, -38.5529480, 30.0827637, -63.9696541, 65.0084839
6: -30.9999695, 30.4651451, -35.3619232, 34.7121735, -65.7121429, 65.8270645
7: -34.0512619, 31.4234161, -38.8661118, 35.7188034, -69.7700653, 70.2895279
8: -45.9617310, 24.8935585, -52.2554207, 28.2796974, -74.2414246, 77.1489792
9: -30.5982361, 30.8947067, -34.8453865, 35.1383247, -65.7365494, 65.7400894

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8458696, upper bound: 132.8517782
time: 10.10 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8500128, upper bound: 132.8517780
time: 10.85 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -48.8449707, 39.0510635, -46.2959442, 37.0286255, -85.8735962, 85.3470001
1: -42.4362144, 34.3010864, -40.2307129, 32.5369148, -74.9731293, 74.5317841
2: -54.4761505, 34.5667953, -51.6183128, 32.7968178, -87.2729645, 86.1851044
3: -57.6259689, 29.7838249, -54.5852165, 28.2878876, -85.9138565, 84.3690414
4: -54.2917328, 39.8984985, -51.4463081, 37.8182907, -92.1100235, 91.3448029
5: -48.0002670, 37.3978653, -45.4896622, 35.4803391, -83.4806061, 82.8875275
6: -44.3113899, 43.4557037, -41.9951401, 41.1989479, -85.5103378, 85.4508362
7: -48.6666451, 43.8947678, -46.0982780, 41.6310997, -90.2977448, 89.9930420
8: -64.0827408, 36.6602554, -60.8202400, 34.7879143, -98.8706360, 97.4804840
9: -43.6560669, 43.9728813, -41.3397255, 41.6908989, -85.3469696, 85.3125992

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8808526, upper bound: 132.8808526
time: 7.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8808526, upper bound: 132.8811433
time: 9.05 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -48.4257736, 38.7211227, -48.9121437, 39.0901871, -87.5159531, 87.6332703
1: -42.0783234, 34.0110664, -42.5127945, 34.3698769, -76.4481964, 76.5238571
2: -54.0117378, 34.2822037, -54.5499992, 34.6543045, -88.6660461, 88.8321991
3: -57.1444206, 29.5254936, -57.7440605, 29.8272953, -86.9717026, 87.2695541
4: -53.8381653, 39.5601196, -54.3758812, 39.9718857, -93.8100510, 93.9360046
5: -47.5893860, 37.0764084, -48.0525208, 37.4278755, -85.0172577, 85.1289291
6: -43.9413376, 43.0823059, -44.4064789, 43.5236664, -87.4650040, 87.4887848
7: -48.2481499, 43.5358658, -48.7374687, 43.9630852, -92.2112350, 92.2733231
8: -63.5585747, 36.3390083, -64.2403107, 36.7102013, -100.2687683, 100.5793152
9: -43.2957458, 43.6030540, -43.6957779, 44.0243797, -87.3201294, 87.2988281

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8811433, upper bound: 132.8808938
time: 7.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8811433, upper bound: 132.8825103
time: 6.87 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -52.2044067, 41.6908340, -46.2959442, 37.0286255, -89.2330246, 87.9867706
1: -45.4122925, 36.6357460, -40.2307129, 32.5369148, -77.9492035, 76.8664474
2: -58.2383804, 36.8959236, -51.6183128, 32.7968178, -91.0352020, 88.5142288
3: -61.6802521, 31.7427750, -54.5852165, 28.2878876, -89.9681320, 86.3279877
4: -58.1083488, 42.5811501, -51.4463081, 37.8182907, -95.9266357, 94.0274582
5: -51.2940598, 39.9740143, -45.4896622, 35.4803391, -86.7743912, 85.4636765
6: -47.3896332, 46.4299850, -41.9951401, 41.1989479, -88.5885696, 88.4251099
7: -52.0556946, 46.9781303, -46.0982780, 41.6310997, -93.6867981, 93.0764084
8: -68.5996933, 38.9452133, -60.8202400, 34.7879143, -103.3876038, 99.7654572
9: -46.6378937, 46.9359703, -41.3397255, 41.6908989, -88.3287964, 88.2756882

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 214

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8596507, upper bound: 132.8613068
time: 8.96 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8652203, upper bound: 132.8655669
time: 8.69 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -51.7847977, 41.3561363, -48.9121437, 39.0901871, -90.8749466, 90.2682800
1: -45.0541077, 36.3422966, -42.5127945, 34.3698769, -79.4239807, 78.8550873
2: -57.7761116, 36.6095734, -54.5499992, 34.6543045, -92.4304199, 91.1595764
3: -61.1963081, 31.4784546, -57.7440605, 29.8272953, -91.0235901, 89.2225189
4: -57.6529770, 42.2423286, -54.3758812, 39.9718857, -97.6248627, 96.6182098
5: -50.8798904, 39.6506424, -48.0525208, 37.4278755, -88.3077698, 87.7031631
6: -47.0136261, 46.0549507, -44.4064789, 43.5236664, -90.5372925, 90.4614258
7: -51.6324577, 46.6202927, -48.7374687, 43.9630852, -95.5955429, 95.3577576
8: -68.0819397, 38.6160049, -64.2403107, 36.7102013, -104.7921448, 102.8563156
9: -46.2697525, 46.5611153, -43.6957779, 44.0243797, -90.2941284, 90.2568970

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8617820, upper bound: 132.8632910
time: 9.95 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8666455, upper bound: 132.8670667
time: 9.07 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -46.8467216, 37.4495277, -51.7301025, 41.3300362, -88.1767578, 89.1796265
1: -40.7592316, 32.9064445, -44.9516335, 36.3198853, -77.0791092, 77.8580627
2: -52.2670593, 33.1602707, -57.6775818, 36.5814552, -88.8485031, 90.8378525
3: -55.2646484, 28.5586796, -61.0941391, 31.5137825, -86.7784271, 89.6528015
4: -52.1219711, 38.2484207, -57.5191574, 42.2138824, -94.3358459, 95.7675629
5: -46.0482635, 35.9077072, -50.8119049, 39.6037598, -85.6520233, 86.7196121
6: -42.4921265, 41.6926956, -46.9621124, 46.0056419, -88.4977722, 88.6548080
7: -46.6595421, 42.1964035, -51.5725136, 46.4818840, -93.1414261, 93.7688980
8: -61.6571045, 35.0534477, -67.8646622, 38.7315369, -100.3886337, 102.9181061
9: -41.8242683, 42.1668777, -46.2219963, 46.5268440, -88.3510895, 88.3888626

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8604437, upper bound: 132.8604068
time: 9.94 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8655669, upper bound: 132.8652203
time: 7.53 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -49.3763428, 39.4441032, -51.3195305, 41.0039978, -90.3803406, 90.7636261
1: -42.9661217, 34.6794891, -44.6011047, 36.0333061, -78.9994125, 79.2805939
2: -55.1016998, 34.9607773, -57.2264633, 36.3017349, -91.4034348, 92.1872253
3: -58.3222275, 30.0463943, -60.6213608, 31.2556591, -89.5778809, 90.6677551
4: -54.9588470, 40.3328896, -57.0732956, 41.8827019, -96.8415451, 97.4061737
5: -48.5285225, 37.7913742, -50.4064140, 39.2880630, -87.8165588, 88.1977844
6: -44.8268890, 43.9401398, -46.5948372, 45.6390648, -90.4659576, 90.5349731
7: -49.2119980, 44.4547462, -51.1582794, 46.1309891, -95.3429871, 95.6130219
8: -64.9677887, 36.9138336, -67.3565598, 38.4119415, -103.3797226, 104.2703857
9: -44.1030426, 44.4269180, -45.8632278, 46.1603394, -90.2633820, 90.2901459

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_A2_B2_A1_A2_A1

### Relational analysis result of IS_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8784782, upper bound: 132.8780408
time: 10.34 seconds

## Relational analysis of IS_A2_B2_A1_A2_A2

### Relational analysis result of IS_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8783791, upper bound: 132.8778454
time: 7.15 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -59.0097885, 47.1433296, -50.9105606, 40.6389999, -99.6487885, 98.0538940
1: -51.1444092, 41.4053345, -44.2342453, 35.6444550, -86.7888641, 85.6395798
2: -65.7771912, 41.6981773, -56.7254066, 35.9721832, -101.7493744, 98.4235840
3: -69.7296829, 35.9015503, -60.0283203, 30.8076897, -100.5373611, 95.9298630
4: -65.5095520, 48.1816788, -56.6944504, 41.4510918, -106.9606476, 104.8761292
5: -57.9485550, 45.0508461, -50.0572205, 38.9541283, -96.9026794, 95.1080627
6: -53.6021080, 52.4646873, -46.1253395, 45.2179985, -98.8201065, 98.5900269
7: -58.8953629, 52.7652702, -50.7492752, 45.8116913, -104.7070465, 103.5145416
8: -76.8466263, 44.4354362, -66.8176422, 37.9100037, -114.7566299, 111.2530670
9: -52.9019966, 53.0811310, -45.5728951, 45.8235245, -98.7255173, 98.6540222

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8753450, upper bound: 132.8759784
time: 10.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8779610, upper bound: 132.8792274
time: 8.26 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 20.33 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8762406, upper bound: 132.8756992
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8762352, upper bound: 132.8756855
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8489408, upper bound: 132.8499579
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8510328, upper bound: 132.8495100
IS_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8461862, upper bound: 132.8459261
IS_A1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8451257, upper bound: 132.8455237
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8759670, upper bound: 132.8753608
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8759670, upper bound: 132.8753608
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8508502, upper bound: 132.8481728
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8508502, upper bound: 132.8481728
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8500128, upper bound: 132.8478292
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8500128, upper bound: 132.8478292
IS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8542066, upper bound: 132.8521347
IS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8542066, upper bound: 132.8521349
IS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8458696, upper bound: 132.8517782
IS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8500128, upper bound: 132.8517780
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8808526, upper bound: 132.8808526
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8808526, upper bound: 132.8811433
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8811433, upper bound: 132.8808938
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8811433, upper bound: 132.8825103
IS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8596507, upper bound: 132.8613068
IS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8652203, upper bound: 132.8655669
IS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8617820, upper bound: 132.8632910
IS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8666455, upper bound: 132.8670667
IS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8604437, upper bound: 132.8604068
IS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8655669, upper bound: 132.8652203
IS_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8784782, upper bound: 132.8780408
IS_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8783791, upper bound: 132.8778454
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8753450, upper bound: 132.8759784
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.33
Output dim: 8, lower bound: -132.8779610, upper bound: 132.8792274
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.33
Output dim: 8, lower bound: -132.8874226, upper bound: 132.8999956
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=145.0349884033203
rel_dist={8: [-132.9262935075506, 132.92629350749155]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1828.12 seconds
