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
execution time: IAR + LP analysis = 1.11 + 10.44 = 11.55 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -132.9266407, upper bound: 132.9266406


# Binary Search by BASE starts (time budget: 2688.45 seconds, max iter: 100)

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
Binary search time: 34.32 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2654.13 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9200525, upper bound: 132.9191463
time: 5.79 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9203334, upper bound: 132.9203334
time: 9.51 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.41 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.41
Output dim: 8, lower bound: -132.9200525, upper bound: 132.9191463
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.41
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

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9190575, upper bound: 132.9190575
time: 6.37 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9190575, upper bound: 132.9191201
time: 5.90 seconds

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
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9191201, upper bound: 132.9200506
time: 6.46 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9191201, upper bound: 132.9203334
time: 9.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.92 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.92
Output dim: 8, lower bound: -132.9190575, upper bound: 132.9190575
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.92
Output dim: 8, lower bound: -132.9190575, upper bound: 132.9191201
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.92
Output dim: 8, lower bound: -132.9191201, upper bound: 132.9200506
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 16.92
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
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9063998, upper bound: 132.9023251
time: 10.76 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9103192, upper bound: 132.9103038
time: 8.94 seconds

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

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9063998, upper bound: 132.9023486
time: 10.79 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9103192, upper bound: 132.9103575
time: 6.19 seconds

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

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9062600, upper bound: 132.9024971
time: 8.47 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9103453, upper bound: 132.9111884
time: 6.40 seconds

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

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9062600, upper bound: 132.9027248
time: 9.41 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9103453, upper bound: 132.9115000
time: 6.86 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 17.40 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.40
Output dim: 8, lower bound: -132.9063998, upper bound: 132.9023251
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.40
Output dim: 8, lower bound: -132.9103192, upper bound: 132.9103038
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.40
Output dim: 8, lower bound: -132.9063998, upper bound: 132.9023486
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.40
Output dim: 8, lower bound: -132.9103192, upper bound: 132.9103575
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.40
Output dim: 8, lower bound: -132.9062600, upper bound: 132.9024971
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.40
Output dim: 8, lower bound: -132.9103453, upper bound: 132.9111884
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.40
Output dim: 8, lower bound: -132.9062600, upper bound: 132.9027248
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.40
Output dim: 8, lower bound: -132.9103453, upper bound: 132.9115000

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -51.6127548, 41.2905121, -64.0446243, 51.2888947, -102.9016418, 105.3351364
1: -44.7630692, 36.1786880, -55.2720528, 45.0459595, -89.8090210, 91.4507446
2: -57.5625687, 36.4666519, -71.3832703, 45.3064804, -102.8690414, 107.8499146
3: -60.8609123, 31.3137283, -75.7214203, 39.1091309, -99.9700470, 107.0351486
4: -57.4186287, 42.1242523, -70.8308105, 52.4435577, -109.8621826, 112.9550476
5: -50.8459854, 39.5425415, -62.9273415, 48.8722038, -99.7181854, 102.4698639
6: -46.7223969, 45.8821716, -58.1959152, 56.9931984, -103.7155914, 104.0780869
7: -51.4561119, 46.3875732, -63.9623642, 56.9240646, -108.3801727, 110.3499222
8: -67.5260162, 38.7139359, -82.6422119, 48.9203262, -116.4463425, 121.3561401
9: -46.3506050, 46.5204468, -57.6472816, 57.6410828, -103.9916840, 104.1677246

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 163

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9009808, upper bound: 132.9009808
time: 6.96 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9009808, upper bound: 132.9024840
time: 7.33 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -60.1992645, 48.1943436, -64.4411926, 51.6097183, -111.8089828, 112.6355362
1: -52.0760002, 42.2844925, -55.6060562, 45.3308105, -97.4068069, 97.8905487
2: -67.1459122, 42.5484734, -71.8256683, 45.5903473, -112.7362595, 114.3741302
3: -71.1507263, 36.6753616, -76.1985168, 39.3602676, -110.5109940, 112.8738708
4: -66.7423553, 49.2569008, -71.2571411, 52.7743759, -119.5167313, 120.5140228
5: -59.2147026, 46.0084000, -63.3112411, 49.1716156, -108.3863220, 109.3196411
6: -54.6522217, 53.5541687, -58.5629044, 57.3490295, -112.0012436, 112.1170731
7: -60.0959549, 53.7517357, -64.3627548, 57.2595749, -117.3555298, 118.1144867
8: -78.1239395, 45.6204758, -83.1233978, 49.2520485, -127.3759842, 128.7438660
9: -54.1358528, 54.1862679, -58.0087967, 57.9970093, -112.1328583, 112.1950684

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9024840, upper bound: 132.9064745
time: 8.20 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9024840, upper bound: 132.9104298
time: 6.60 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -51.6127548, 41.2905121, -68.3980026, 54.7145958, -106.3273468, 109.6885147
1: -44.7630692, 36.1786880, -59.0821266, 48.1056366, -92.8686981, 95.2608185
2: -57.5625687, 36.4666519, -76.2914124, 48.3526611, -105.9152222, 112.7580566
3: -60.8609123, 31.3137283, -80.9886780, 41.6674347, -102.5283508, 112.3024063
4: -57.4186287, 42.1242523, -75.7454910, 55.9742432, -113.3928680, 117.8697357
5: -50.8459854, 39.5425415, -67.2117081, 52.1692505, -103.0152359, 106.7542343
6: -46.7223969, 45.8821716, -62.1522980, 60.8865242, -107.6089096, 108.0344696
7: -51.4561119, 46.3875732, -68.3723297, 60.8551140, -112.3112259, 114.7599030
8: -67.5260162, 38.7139359, -88.3538055, 52.0084572, -119.5344696, 127.0677414
9: -46.3506050, 46.5204468, -61.5536880, 61.5289650, -107.8795700, 108.0741272

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9010549, upper bound: 132.9007671
time: 7.76 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9010549, upper bound: 132.9023486
time: 6.57 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -60.1992645, 48.1943436, -68.7814865, 55.0239449, -115.2232056, 116.9758301
1: -52.0760002, 42.2844925, -59.4057541, 48.3804512, -100.4564514, 101.6902390
2: -67.1459122, 42.5484734, -76.7185211, 48.6273499, -115.7732620, 119.2669830
3: -71.1507263, 36.6753616, -81.4480972, 41.9100647, -113.0607758, 118.1234589
4: -66.7423553, 49.2569008, -76.1565170, 56.2944107, -123.0367661, 125.4134064
5: -59.2147026, 46.0084000, -67.5821457, 52.4586334, -111.6733170, 113.5905457
6: -54.6522217, 53.5541687, -62.5059471, 61.2303734, -115.8825989, 116.0601196
7: -60.0959549, 53.7517357, -68.7584686, 61.1786499, -121.2745895, 122.5102005
8: -78.1239395, 45.6204758, -88.8186417, 52.3298111, -130.4537354, 134.4391174
9: -54.1358528, 54.1862679, -61.9029617, 61.8735733, -116.0094147, 116.0892181

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9024996, upper bound: 132.9062734
time: 8.30 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9024996, upper bound: 132.9103575
time: 8.05 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -55.8258896, 44.6119957, -64.0446243, 51.2888947, -107.1147842, 108.6566162
1: -48.4584999, 39.1351471, -55.2720528, 45.0459595, -93.5044479, 94.4071960
2: -62.3182716, 39.4010353, -71.3832703, 45.3064804, -107.6247559, 110.7843018
3: -65.9603882, 33.7978401, -75.7214203, 39.1091309, -105.0695114, 109.5192490
4: -62.1853561, 45.5433121, -70.8308105, 52.4435577, -114.6289139, 116.3741226
5: -54.9995155, 42.7348557, -62.9273415, 48.8722038, -103.8717194, 105.6621933
6: -50.5469513, 49.6461296, -58.1959152, 56.9931984, -107.5401459, 107.8420410
7: -55.7218704, 50.2024231, -63.9623642, 56.9240646, -112.6459351, 114.1647873
8: -73.0749130, 41.6707306, -82.6422119, 48.9203262, -121.9952393, 124.3129272
9: -50.1233025, 50.2728882, -57.6472816, 57.6410828, -107.7643890, 107.9201584

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 163

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9007671, upper bound: 132.9010549
time: 6.40 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9007671, upper bound: 132.9024996
time: 7.90 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -64.2413177, 51.3793373, -64.4411926, 51.6097183, -115.8510361, 115.8205261
1: -55.6276360, 45.1248512, -55.6060562, 45.3308105, -100.9584503, 100.7309113
2: -71.7098541, 45.3791008, -71.8256683, 45.5903473, -117.3002014, 117.2047653
3: -76.0516815, 39.0437317, -76.1985168, 39.3602676, -115.4119492, 115.2422333
4: -71.3212051, 52.5394173, -71.2571411, 52.7743759, -124.0955811, 123.7965546
5: -63.2045441, 49.0727196, -63.3112411, 49.1716156, -112.3761520, 112.3839569
6: -58.3308105, 57.1740952, -58.5629044, 57.3490295, -115.6798325, 115.7369919
7: -64.1979294, 57.4201889, -64.3627548, 57.2595749, -121.4574890, 121.7829437
8: -83.4566879, 48.4633865, -83.1233978, 49.2520485, -132.7087250, 131.5867920
9: -57.7637482, 57.7954025, -58.0087967, 57.9970093, -115.7607574, 115.8041992

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9023486, upper bound: 132.9070402
time: 8.05 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9023486, upper bound: 132.9070402
time: 8.25 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -55.8258896, 44.6119957, -68.3980026, 54.7145958, -110.5404816, 113.0099945
1: -48.4584999, 39.1351471, -59.0821266, 48.1056366, -96.5641327, 98.2172699
2: -62.3182716, 39.4010353, -76.2914124, 48.3526611, -110.6709290, 115.6924438
3: -65.9603882, 33.7978401, -80.9886780, 41.6674347, -107.6278076, 114.7865143
4: -62.1853561, 45.5433121, -75.7454910, 55.9742432, -118.1595840, 121.2888031
5: -54.9995155, 42.7348557, -67.2117081, 52.1692505, -107.1687622, 109.9465637
6: -50.5469513, 49.6461296, -62.1522980, 60.8865242, -111.4334717, 111.7984161
7: -55.7218704, 50.2024231, -68.3723297, 60.8551140, -116.5769806, 118.5747528
8: -73.0749130, 41.6707306, -88.3538055, 52.0084572, -125.0833740, 130.0245361
9: -50.1233025, 50.2728882, -61.5536880, 61.5289650, -111.6522675, 111.8265533

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9012626, upper bound: 132.9012767
time: 7.55 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9012626, upper bound: 132.9027248
time: 6.26 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -64.2413177, 51.3793373, -68.7814865, 55.0239449, -119.2652588, 120.1608124
1: -55.6276360, 45.1248512, -59.4057541, 48.3804512, -104.0080872, 104.5305939
2: -71.7098541, 45.3791008, -76.7185211, 48.6273499, -120.3372040, 122.0976181
3: -76.0516815, 39.0437317, -81.4480972, 41.9100647, -117.9617386, 120.4918289
4: -71.3212051, 52.5394173, -76.1565170, 56.2944107, -127.6156082, 128.6959381
5: -63.2045441, 49.0727196, -67.5821457, 52.4586334, -115.6631622, 116.6548615
6: -58.3308105, 57.1740952, -62.5059471, 61.2303734, -119.5611877, 119.6800308
7: -64.1979294, 57.4201889, -68.7584686, 61.1786499, -125.3765793, 126.1786575
8: -83.4566879, 48.4633865, -88.8186417, 52.3298111, -135.7864838, 137.2820282
9: -57.7637482, 57.7954025, -61.9029617, 61.8735733, -119.6373138, 119.6983566

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9027153, upper bound: 132.9072008
time: 7.56 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9027153, upper bound: 132.9115000
time: 8.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 17.38 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.38
Output dim: 8, lower bound: -132.9009808, upper bound: 132.9009808
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.38
Output dim: 8, lower bound: -132.9009808, upper bound: 132.9024840
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.38
Output dim: 8, lower bound: -132.9024840, upper bound: 132.9064745
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.38
Output dim: 8, lower bound: -132.9024840, upper bound: 132.9104298
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.38
Output dim: 8, lower bound: -132.9010549, upper bound: 132.9007671
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.38
Output dim: 8, lower bound: -132.9010549, upper bound: 132.9023486
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.38
Output dim: 8, lower bound: -132.9024996, upper bound: 132.9062734
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.38
Output dim: 8, lower bound: -132.9024996, upper bound: 132.9103575
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.38
Output dim: 8, lower bound: -132.9007671, upper bound: 132.9010549
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.38
Output dim: 8, lower bound: -132.9007671, upper bound: 132.9024996
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.38
Output dim: 8, lower bound: -132.9023486, upper bound: 132.9070402
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.38
Output dim: 8, lower bound: -132.9023486, upper bound: 132.9070402
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.38
Output dim: 8, lower bound: -132.9012626, upper bound: 132.9012767
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.38
Output dim: 8, lower bound: -132.9012626, upper bound: 132.9027248
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.38
Output dim: 8, lower bound: -132.9027153, upper bound: 132.9072008
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.38
Output dim: 8, lower bound: -132.9027153, upper bound: 132.9115000

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -51.6127548, 41.2905121, -51.6127548, 41.2905121, -92.9032669, 92.9032669
1: -44.7630692, 36.1786880, -44.7630692, 36.1786880, -80.9417496, 80.9417419
2: -57.5625687, 36.4666519, -57.5625687, 36.4666519, -94.0292130, 94.0292130
3: -60.8609123, 31.3137283, -60.8609123, 31.3137283, -92.1746368, 92.1746368
4: -57.4186287, 42.1242523, -57.4186287, 42.1242523, -99.5428772, 99.5428772
5: -50.8459854, 39.5425415, -50.8459854, 39.5425415, -90.3885193, 90.3885117
6: -46.7223969, 45.8821716, -46.7223969, 45.8821716, -92.6045685, 92.6045685
7: -51.4561119, 46.3875732, -51.4561119, 46.3875732, -97.8436890, 97.8436890
8: -67.5260162, 38.7139359, -67.5260162, 38.7139359, -106.2399445, 106.2399521
9: -46.3506050, 46.5204468, -46.3506050, 46.5204468, -92.8710480, 92.8710480

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8727582, upper bound: 132.8763299
time: 8.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8703524, upper bound: 132.8703524
time: 5.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -51.6127548, 41.2905121, -60.1992645, 48.1943436, -99.8070984, 101.4897766
1: -44.7630692, 36.1786880, -52.0760002, 42.2844925, -87.0475540, 88.2546844
2: -57.5625687, 36.4666519, -67.1459122, 42.5484734, -100.1110229, 103.6125641
3: -60.8609123, 31.3137283, -71.1507263, 36.6753616, -97.5362701, 102.4644547
4: -57.4186287, 42.1242523, -66.7423553, 49.2569008, -106.6755295, 108.8666000
5: -50.8459854, 39.5425415, -59.2147026, 46.0084000, -96.8543854, 98.7572250
6: -46.7223969, 45.8821716, -54.6522217, 53.5541687, -100.2765656, 100.5343933
7: -51.4561119, 46.3875732, -60.0959549, 53.7517357, -105.2078400, 106.4835205
8: -67.5260162, 38.7139359, -78.1239395, 45.6204758, -113.1464920, 116.8378754
9: -46.3506050, 46.5204468, -54.1358528, 54.1862679, -100.5368729, 100.6562958

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8727582, upper bound: 132.8784746
time: 8.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8703524, upper bound: 132.8723385
time: 6.32 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -60.1992645, 48.1943436, -51.6127548, 41.2905121, -101.4897766, 99.8070984
1: -52.0760002, 42.2844925, -44.7630692, 36.1786880, -88.2546844, 87.0475464
2: -67.1459122, 42.5484734, -57.5625687, 36.4666519, -103.6125641, 100.1110229
3: -71.1507263, 36.6753616, -60.8609123, 31.3137283, -102.4644547, 97.5362701
4: -66.7423553, 49.2569008, -57.4186287, 42.1242523, -108.8666077, 106.6755295
5: -59.2147026, 46.0084000, -50.8459854, 39.5425415, -98.7572174, 96.8543854
6: -54.6522217, 53.5541687, -46.7223969, 45.8821716, -100.5343933, 100.2765656
7: -60.0959549, 53.7517357, -51.4561119, 46.3875732, -106.4835205, 105.2078400
8: -78.1239395, 45.6204758, -67.5260162, 38.7139359, -116.8378754, 113.1464920
9: -54.1358528, 54.1862679, -46.3506050, 46.5204468, -100.6562958, 100.5368729

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8744029, upper bound: 132.8820877
time: 6.46 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8723385, upper bound: 132.8778062
time: 6.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -60.1992645, 48.1943436, -60.1992645, 48.1943436, -108.3936081, 108.3936081
1: -52.0760002, 42.2844925, -52.0760002, 42.2844925, -94.3604889, 94.3604889
2: -67.1459122, 42.5484734, -67.1459122, 42.5484734, -109.6943741, 109.6943741
3: -71.1507263, 36.6753616, -71.1507263, 36.6753616, -107.8260880, 107.8260880
4: -66.7423553, 49.2569008, -66.7423553, 49.2569008, -115.9992523, 115.9992447
5: -59.2147026, 46.0084000, -59.2147026, 46.0084000, -105.2230988, 105.2230988
6: -54.6522217, 53.5541687, -54.6522217, 53.5541687, -108.2063904, 108.2063904
7: -60.0959549, 53.7517357, -60.0959549, 53.7517357, -113.8476868, 113.8476868
8: -78.1239395, 45.6204758, -78.1239395, 45.6204758, -123.7444153, 123.7444153
9: -54.1358528, 54.1862679, -54.1358528, 54.1862679, -108.3221130, 108.3221130

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8744029, upper bound: 132.8892079
time: 7.49 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8723385, upper bound: 132.8778063
time: 7.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -51.6127548, 41.2905121, -55.8258896, 44.6119957, -96.2247467, 97.1164017
1: -44.7630692, 36.1786880, -48.4584999, 39.1351471, -83.8982086, 84.6371841
2: -57.5625687, 36.4666519, -62.3182716, 39.4010353, -96.9636002, 98.7849274
3: -60.8609123, 31.3137283, -65.9603882, 33.7978401, -94.6587524, 97.2741165
4: -57.4186287, 42.1242523, -62.1853561, 45.5433121, -102.9619446, 104.3095932
5: -50.8459854, 39.5425415, -54.9995155, 42.7348557, -93.5808411, 94.5420532
6: -46.7223969, 45.8821716, -50.5469513, 49.6461296, -96.3685150, 96.4291229
7: -51.4561119, 46.3875732, -55.7218704, 50.2024231, -101.6585388, 102.1094437
8: -67.5260162, 38.7139359, -73.0749130, 41.6707306, -109.1967392, 111.7888412
9: -46.3506050, 46.5204468, -50.1233025, 50.2728882, -96.6234894, 96.6437531

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8727346, upper bound: 132.8759142
time: 7.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8703766, upper bound: 132.8701018
time: 7.30 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -51.6127548, 41.2905121, -64.2413177, 51.3793373, -102.9920807, 105.5318298
1: -44.7630692, 36.1786880, -55.6276360, 45.1248512, -89.8879013, 91.8063202
2: -57.5625687, 36.4666519, -71.7098541, 45.3791008, -102.9416580, 108.1765060
3: -60.8609123, 31.3137283, -76.0516815, 39.0437317, -99.9046478, 107.3654099
4: -57.4186287, 42.1242523, -71.3212051, 52.5394173, -109.9580460, 113.4454575
5: -50.8459854, 39.5425415, -63.2045441, 49.0727196, -99.9187012, 102.7470703
6: -46.7223969, 45.8821716, -58.3308105, 57.1740952, -103.8964767, 104.2129822
7: -51.4561119, 46.3875732, -64.1979294, 57.4201889, -108.8762970, 110.5854950
8: -67.5260162, 38.7139359, -83.4566879, 48.4633865, -115.9894028, 122.1706238
9: -46.3506050, 46.5204468, -57.7637482, 57.7954025, -104.1460114, 104.2841949

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8727346, upper bound: 132.8759142
time: 8.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8703766, upper bound: 132.8722143
time: 7.04 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -60.1992645, 48.1943436, -55.8258896, 44.6119957, -104.8112640, 104.0202332
1: -52.0760002, 42.2844925, -48.4584999, 39.1351471, -91.2111511, 90.7429886
2: -67.1459122, 42.5484734, -62.3182716, 39.4010353, -106.5469513, 104.8667374
3: -71.1507263, 36.6753616, -65.9603882, 33.7978401, -104.9485550, 102.6357498
4: -66.7423553, 49.2569008, -62.1853561, 45.5433121, -112.2856674, 111.4422379
5: -59.2147026, 46.0084000, -54.9995155, 42.7348557, -101.9495544, 101.0079193
6: -54.6522217, 53.5541687, -50.5469513, 49.6461296, -104.2983398, 104.1011200
7: -60.0959549, 53.7517357, -55.7218704, 50.2024231, -110.2983780, 109.4736023
8: -78.1239395, 45.6204758, -73.0749130, 41.6707306, -119.7946548, 118.6953888
9: -54.1358528, 54.1862679, -50.1233025, 50.2728882, -104.4087296, 104.3095703

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8742351, upper bound: 132.8816989
time: 14.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8722390, upper bound: 132.8774918
time: 7.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -60.1992645, 48.1943436, -64.2413177, 51.3793373, -111.5785980, 112.4356613
1: -52.0760002, 42.2844925, -55.6276360, 45.1248512, -97.2008438, 97.9121246
2: -67.1459122, 42.5484734, -71.7098541, 45.3791008, -112.5250092, 114.2583160
3: -71.1507263, 36.6753616, -76.0516815, 39.0437317, -110.1944504, 112.7270432
4: -66.7423553, 49.2569008, -71.3212051, 52.5394173, -119.2817688, 120.5781097
5: -59.2147026, 46.0084000, -63.2045441, 49.0727196, -108.2874222, 109.2129288
6: -54.6522217, 53.5541687, -58.3308105, 57.1740952, -111.8263092, 111.8849792
7: -60.0959549, 53.7517357, -64.1979294, 57.4201889, -117.5161438, 117.9496613
8: -78.1239395, 45.6204758, -83.4566879, 48.4633865, -126.5873260, 129.0771637
9: -54.1358528, 54.1862679, -57.7637482, 57.7954025, -111.9312592, 111.9500122

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8742351, upper bound: 132.8889642
time: 7.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8722390, upper bound: 132.8774918
time: 7.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -55.8258896, 44.6119957, -51.6127548, 41.2905121, -97.1164017, 96.2247467
1: -48.4584999, 39.1351471, -44.7630692, 36.1786880, -84.6371841, 83.8982086
2: -62.3182716, 39.4010353, -57.5625687, 36.4666519, -98.7849274, 96.9636002
3: -65.9603882, 33.7978401, -60.8609123, 31.3137283, -97.2741165, 94.6587524
4: -62.1853561, 45.5433121, -57.4186287, 42.1242523, -104.3096008, 102.9619446
5: -54.9995155, 42.7348557, -50.8459854, 39.5425415, -94.5420532, 93.5808411
6: -50.5469513, 49.6461296, -46.7223969, 45.8821716, -96.4291229, 96.3685150
7: -55.7218704, 50.2024231, -51.4561119, 46.3875732, -102.1094437, 101.6585388
8: -73.0749130, 41.6707306, -67.5260162, 38.7139359, -111.7888489, 109.1967316
9: -50.1233025, 50.2728882, -46.3506050, 46.5204468, -96.6437531, 96.6234894

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8724913, upper bound: 132.8763022
time: 7.12 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8701018, upper bound: 132.8703766
time: 6.91 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -55.8258896, 44.6119957, -60.1992645, 48.1943436, -104.0202332, 104.8112640
1: -48.4584999, 39.1351471, -52.0760002, 42.2844925, -90.7429886, 91.2111511
2: -62.3182716, 39.4010353, -67.1459122, 42.5484734, -104.8667450, 106.5469513
3: -65.9603882, 33.7978401, -71.1507263, 36.6753616, -102.6357498, 104.9485550
4: -62.1853561, 45.5433121, -66.7423553, 49.2569008, -111.4422455, 112.2856674
5: -54.9995155, 42.7348557, -59.2147026, 46.0084000, -101.0079193, 101.9495544
6: -50.5469513, 49.6461296, -54.6522217, 53.5541687, -104.1011200, 104.2983398
7: -55.7218704, 50.2024231, -60.0959549, 53.7517357, -109.4736023, 110.2983780
8: -73.0749130, 41.6707306, -78.1239395, 45.6204758, -118.6953888, 119.7946625
9: -50.1233025, 50.2728882, -54.1358528, 54.1862679, -104.3095703, 104.4087296

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8724913, upper bound: 132.8763022
time: 8.44 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8701018, upper bound: 132.8722390
time: 7.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -64.2413177, 51.3793373, -51.6127548, 41.2905121, -105.5318298, 102.9920883
1: -55.6276360, 45.1248512, -44.7630692, 36.1786880, -91.8063202, 89.8879089
2: -71.7098541, 45.3791008, -57.5625687, 36.4666519, -108.1765060, 102.9416656
3: -76.0516815, 39.0437317, -60.8609123, 31.3137283, -107.3654099, 99.9046478
4: -71.3212051, 52.5394173, -57.4186287, 42.1242523, -113.4454498, 109.9580460
5: -63.2045441, 49.0727196, -50.8459854, 39.5425415, -102.7470703, 99.9187012
6: -58.3308105, 57.1740952, -46.7223969, 45.8821716, -104.2129822, 103.8964767
7: -64.1979294, 57.4201889, -51.4561119, 46.3875732, -110.5854874, 108.8762970
8: -83.4566879, 48.4633865, -67.5260162, 38.7139359, -122.1706085, 115.9894028
9: -57.7637482, 57.7954025, -46.3506050, 46.5204468, -104.2841949, 104.1460114

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8742363, upper bound: 132.8825568
time: 8.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8722143, upper bound: 132.8781715
time: 7.46 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -64.2413177, 51.3793373, -60.1992645, 48.1943436, -112.4356613, 111.5785980
1: -55.6276360, 45.1248512, -52.0760002, 42.2844925, -97.9121246, 97.2008438
2: -71.7098541, 45.3791008, -67.1459122, 42.5484734, -114.2583160, 112.5250092
3: -76.0516815, 39.0437317, -71.1507263, 36.6753616, -112.7270432, 110.1944580
4: -71.3212051, 52.5394173, -66.7423553, 49.2569008, -120.5780945, 119.2817688
5: -63.2045441, 49.0727196, -59.2147026, 46.0084000, -109.2129364, 108.2874222
6: -58.3308105, 57.1740952, -54.6522217, 53.5541687, -111.8849792, 111.8263092
7: -64.1979294, 57.4201889, -60.0959549, 53.7517357, -117.9496613, 117.5161438
8: -83.4566879, 48.4633865, -78.1239395, 45.6204758, -129.0771637, 126.5873260
9: -57.7637482, 57.7954025, -54.1358528, 54.1862679, -111.9500122, 111.9312515

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8742363, upper bound: 132.8898026
time: 8.29 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8722143, upper bound: 132.8781715
time: 8.26 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -55.8258896, 44.6119957, -55.8258896, 44.6119957, -100.4378815, 100.4378815
1: -48.4584999, 39.1351471, -48.4584999, 39.1351471, -87.5936432, 87.5936432
2: -62.3182716, 39.4010353, -62.3182716, 39.4010353, -101.7193069, 101.7193069
3: -65.9603882, 33.7978401, -65.9603882, 33.7978401, -99.7582169, 99.7582169
4: -62.1853561, 45.5433121, -62.1853561, 45.5433121, -107.7286682, 107.7286606
5: -54.9995155, 42.7348557, -54.9995155, 42.7348557, -97.7343750, 97.7343750
6: -50.5469513, 49.6461296, -50.5469513, 49.6461296, -100.1930847, 100.1930847
7: -55.7218704, 50.2024231, -55.7218704, 50.2024231, -105.9242935, 105.9242935
8: -73.0749130, 41.6707306, -73.0749130, 41.6707306, -114.7456284, 114.7456284
9: -50.1233025, 50.2728882, -50.1233025, 50.2728882, -100.3961868, 100.3961868

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8728386, upper bound: 132.8763512
time: 8.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8705698, upper bound: 132.8705704
time: 7.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -55.8258896, 44.6119957, -64.2413177, 51.3793373, -107.2052307, 108.8533173
1: -48.4584999, 39.1351471, -55.6276360, 45.1248512, -93.5833435, 94.7627869
2: -62.3182716, 39.4010353, -71.7098541, 45.3791008, -107.6973724, 111.1108856
3: -65.9603882, 33.7978401, -76.0516815, 39.0437317, -105.0041122, 109.8495102
4: -62.1853561, 45.5433121, -71.3212051, 52.5394173, -114.7247696, 116.8645172
5: -54.9995155, 42.7348557, -63.2045441, 49.0727196, -104.0722351, 105.9393997
6: -50.5469513, 49.6461296, -58.3308105, 57.1740952, -107.7210388, 107.9769363
7: -55.7218704, 50.2024231, -64.1979294, 57.4201889, -113.1420593, 114.4003525
8: -73.0749130, 41.6707306, -83.4566879, 48.4633865, -121.5382996, 125.1273956
9: -50.1233025, 50.2728882, -57.7637482, 57.7954025, -107.9187012, 108.0366287

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8728386, upper bound: 132.8786910
time: 6.73 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8705698, upper bound: 132.8724641
time: 6.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -64.2413177, 51.3793373, -55.8258896, 44.6119957, -108.8533173, 107.2052307
1: -55.6276360, 45.1248512, -48.4584999, 39.1351471, -94.7627869, 93.5833435
2: -71.7098541, 45.3791008, -62.3182716, 39.4010353, -111.1108856, 107.6973724
3: -76.0516815, 39.0437317, -65.9603882, 33.7978401, -109.8495102, 105.0041199
4: -71.3212051, 52.5394173, -62.1853561, 45.5433121, -116.8645172, 114.7247696
5: -63.2045441, 49.0727196, -54.9995155, 42.7348557, -105.9393921, 104.0722351
6: -58.3308105, 57.1740952, -50.5469513, 49.6461296, -107.9769363, 107.7210388
7: -64.1979294, 57.4201889, -55.7218704, 50.2024231, -114.4003525, 113.1420593
8: -83.4566879, 48.4633865, -73.0749130, 41.6707306, -125.1274109, 121.5382996
9: -57.7637482, 57.7954025, -50.1233025, 50.2728882, -108.0366287, 107.9187012

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8744109, upper bound: 132.8825836
time: 7.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8724674, upper bound: 132.8782397
time: 7.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -64.2413177, 51.3793373, -64.2413177, 51.3793373, -115.6206512, 115.6206512
1: -55.6276360, 45.1248512, -55.6276360, 45.1248512, -100.7524872, 100.7524872
2: -71.7098541, 45.3791008, -71.7098541, 45.3791008, -117.0889511, 117.0889587
3: -76.0516815, 39.0437317, -76.0516815, 39.0437317, -115.0954132, 115.0954132
4: -71.3212051, 52.5394173, -71.3212051, 52.5394173, -123.8606262, 123.8606262
5: -63.2045441, 49.0727196, -63.2045441, 49.0727196, -112.2772522, 112.2772522
6: -58.3308105, 57.1740952, -58.3308105, 57.1740952, -115.5048981, 115.5048981
7: -64.1979294, 57.4201889, -64.1979294, 57.4201889, -121.6181183, 121.6181183
8: -83.4566879, 48.4633865, -83.4566879, 48.4633865, -131.9200745, 131.9200745
9: -57.7637482, 57.7954025, -57.7637482, 57.7954025, -115.5591507, 115.5591507

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8744109, upper bound: 132.8825836
time: 10.20 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8724674, upper bound: 132.8844167
time: 9.48 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 20.82 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8727582, upper bound: 132.8763299
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8703524, upper bound: 132.8703524
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8727582, upper bound: 132.8784746
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8703524, upper bound: 132.8723385
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8744029, upper bound: 132.8820877
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8723385, upper bound: 132.8778062
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8744029, upper bound: 132.8892079
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8723385, upper bound: 132.8778063
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8727346, upper bound: 132.8759142
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8703766, upper bound: 132.8701018
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8727346, upper bound: 132.8759142
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8703766, upper bound: 132.8722143
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8742351, upper bound: 132.8816989
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8722390, upper bound: 132.8774918
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8742351, upper bound: 132.8889642
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8722390, upper bound: 132.8774918
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8724913, upper bound: 132.8763022
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8701018, upper bound: 132.8703766
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8724913, upper bound: 132.8763022
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8701018, upper bound: 132.8722390
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8742363, upper bound: 132.8825568
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8722143, upper bound: 132.8781715
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8742363, upper bound: 132.8898026
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8722143, upper bound: 132.8781715
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8728386, upper bound: 132.8763512
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8705698, upper bound: 132.8705704
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8728386, upper bound: 132.8786910
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8705698, upper bound: 132.8724641
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8744109, upper bound: 132.8825836
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8724674, upper bound: 132.8782397
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8744109, upper bound: 132.8825836
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.82
Output dim: 8, lower bound: -132.8724674, upper bound: 132.8844167

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -45.4646072, 36.3738899, -51.6127548, 41.2905121, -86.7551193, 87.9866409
1: -39.5530090, 31.8691483, -44.7630692, 36.1786880, -75.7316971, 76.6322021
2: -50.7160339, 32.1073608, -57.5625687, 36.4666519, -87.1826859, 89.6699219
3: -53.5997238, 27.4769497, -60.8609123, 31.3137283, -84.9134521, 88.3378601
4: -50.7307816, 37.0094910, -57.4186287, 42.1242523, -92.8550339, 94.4281158
5: -44.8536415, 34.9422836, -50.8459854, 39.5425415, -84.3961639, 85.7882690
6: -41.0696716, 40.4150887, -46.7223969, 45.8821716, -86.9518433, 87.1374817
7: -45.2606277, 41.1597824, -51.4561119, 46.3875732, -91.6481934, 92.6158905
8: -60.0370941, 33.6538963, -67.5260162, 38.7139359, -98.7510300, 101.1799164
9: -40.7661743, 40.9948769, -46.3506050, 46.5204468, -87.2866211, 87.3454819

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8702742, upper bound: 132.8702742
time: 6.47 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8702742, upper bound: 132.8702742
time: 6.99 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -42.3403854, 33.8333168, -48.5308571, 38.8275642, -81.1679535, 82.3641739
1: -37.0862236, 29.6610527, -42.1589622, 34.0146713, -71.1008759, 71.8200150
2: -47.3009338, 29.8201599, -54.1365585, 34.2841568, -81.5850830, 83.9567184
3: -49.9999390, 25.4085732, -57.2200279, 29.3962021, -79.3961334, 82.6286011
4: -47.5493851, 34.3261909, -54.0713577, 39.5669060, -87.1162872, 88.3975525
5: -41.8449402, 32.6362076, -47.8471413, 37.2397728, -79.0847092, 80.4833450
6: -38.1907196, 37.6820030, -43.8965607, 43.1447983, -81.3355179, 81.5785675
7: -42.1527023, 38.7277756, -48.3628159, 43.7714348, -85.9241333, 87.0905685
8: -56.6202621, 30.6124821, -63.7741852, 36.1795845, -92.7998352, 94.3866653
9: -37.8412018, 38.0988007, -43.5420494, 43.7452393, -81.5864410, 81.6408386

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8500389, upper bound: 132.8449517
time: 9.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8409191, upper bound: 132.8409191
time: 5.26 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -45.4646072, 36.3738899, -60.1992645, 48.1943436, -93.6589508, 96.5731506
1: -39.5530090, 31.8691483, -52.0760002, 42.2844925, -81.8375015, 83.9451447
2: -50.7160339, 32.1073608, -67.1459122, 42.5484734, -93.2644958, 99.2532654
3: -53.5997238, 27.4769497, -71.1507263, 36.6753616, -90.2750854, 98.6276779
4: -50.7307816, 37.0094910, -66.7423553, 49.2569008, -99.9876862, 103.7518387
5: -44.8536415, 34.9422836, -59.2147026, 46.0084000, -90.8620453, 94.1569824
6: -41.0696716, 40.4150887, -54.6522217, 53.5541687, -94.6238403, 95.0672989
7: -45.2606277, 41.1597824, -60.0959549, 53.7517357, -99.0123596, 101.2557373
8: -60.0370941, 33.6538963, -78.1239395, 45.6204758, -105.6575699, 111.7778320
9: -40.7661743, 40.9948769, -54.1358528, 54.1862679, -94.9524384, 95.1307144

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8777305, upper bound: 132.8722895
time: 9.28 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8777305, upper bound: 132.8722895
time: 9.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -42.3403854, 33.8333168, -56.7182274, 45.4052353, -87.7456131, 90.5515442
1: -37.0862236, 29.6610527, -49.1242638, 39.8117104, -76.8979111, 78.7853165
2: -47.3009338, 29.8201599, -63.2660599, 40.0583725, -87.3593063, 93.0862198
3: -49.9999390, 25.4085732, -67.0265579, 34.5253830, -84.5253067, 92.4351349
4: -47.5493851, 34.3261909, -62.9607353, 46.3629723, -93.9123535, 97.2869263
5: -41.8449402, 32.6362076, -55.8210144, 43.4001808, -85.2451019, 88.4572144
6: -38.1907196, 37.6820030, -51.4424477, 50.4453735, -88.6360855, 89.1244507
7: -42.1527023, 38.7277756, -56.5832443, 50.7898293, -92.9425201, 95.3110199
8: -56.6202621, 30.6124821, -73.9002075, 42.7544327, -99.3746948, 104.5126877
9: -37.8412018, 38.0988007, -50.9484138, 51.0501518, -88.8913422, 89.0472031

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8777305, upper bound: 132.8723369
time: 8.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8777305, upper bound: 132.8723385
time: 8.90 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -53.0825882, 42.4911690, -51.6127548, 41.2905121, -94.3731003, 94.1038971
1: -46.0360680, 37.2525978, -44.7630692, 36.1786880, -82.2147522, 82.0156555
2: -59.2202911, 37.4693909, -57.5625687, 36.4666519, -95.6869354, 95.0319595
3: -62.7199554, 32.2596054, -60.8609123, 31.3137283, -94.0336838, 93.1205139
4: -59.0259171, 43.3376617, -57.4186287, 42.1242523, -101.1501617, 100.7562866
5: -52.2700615, 40.6751938, -50.8459854, 39.5425415, -91.8126068, 91.5211792
6: -48.0805702, 47.2115555, -46.7223969, 45.8821716, -93.9627380, 93.9339523
7: -52.9167786, 47.7058334, -51.4561119, 46.3875732, -99.3043518, 99.1619415
8: -69.4956894, 39.7534752, -67.5260162, 38.7139359, -108.2096252, 107.2794952
9: -47.6412315, 47.7995987, -46.3506050, 46.5204468, -94.1616821, 94.1501923

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8722895, upper bound: 132.8777305
time: 7.04 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8722895, upper bound: 132.8777305
time: 7.18 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 15.37 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.37
Output dim: 8, lower bound: -132.8702742, upper bound: 132.8702742
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.37
Output dim: 8, lower bound: -132.8702742, upper bound: 132.8702742
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.37
Output dim: 8, lower bound: -132.8500389, upper bound: 132.8449517
IS_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 15.37
Output dim: 8, lower bound: -132.8409191, upper bound: 132.8409191
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.37
Output dim: 8, lower bound: -132.8777305, upper bound: 132.8722895
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.37
Output dim: 8, lower bound: -132.8777305, upper bound: 132.8722895
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.37
Output dim: 8, lower bound: -132.8777305, upper bound: 132.8723369
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.37
Output dim: 8, lower bound: -132.8777305, upper bound: 132.8723385
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.37
Output dim: 8, lower bound: -132.8722895, upper bound: 132.8777305
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.37
Output dim: 8, lower bound: -132.8722895, upper bound: 132.8777305
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8723385, upper bound: 132.8778062
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8744029, upper bound: 132.8892079
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8723385, upper bound: 132.8778063
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8727346, upper bound: 132.8759142
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8703766, upper bound: 132.8701018
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8727346, upper bound: 132.8759142
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8703766, upper bound: 132.8722143
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8742351, upper bound: 132.8816989
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8722390, upper bound: 132.8774918
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8742351, upper bound: 132.8889642
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8722390, upper bound: 132.8774918
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8724913, upper bound: 132.8763022
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8701018, upper bound: 132.8703766
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8724913, upper bound: 132.8763022
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8701018, upper bound: 132.8722390
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8742363, upper bound: 132.8825568
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8722143, upper bound: 132.8781715
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8742363, upper bound: 132.8898026
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8722143, upper bound: 132.8781715
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8728386, upper bound: 132.8763512
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8705698, upper bound: 132.8705704
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8728386, upper bound: 132.8786910
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8705698, upper bound: 132.8724641
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8744109, upper bound: 132.8825836
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8724674, upper bound: 132.8782397
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8744109, upper bound: 132.8825836
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.37
Output dim: 8, lower bound: -132.8724674, upper bound: 132.8844167
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=145.0349884033203
rel_dist={8: [-132.92649402057955, 132.9264940353584]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9197241, upper bound: 132.9190470
time: 9.70 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9202565, upper bound: 132.9202565
time: 7.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.39 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.39
Output dim: 8, lower bound: -132.9197241, upper bound: 132.9190470
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.39
Output dim: 8, lower bound: -132.9202565, upper bound: 132.9202565

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -65.2420654, 52.2499275, -69.0227661, 55.2618904, -120.5039520, 121.2726898
1: -56.2837639, 45.9009018, -59.4661903, 48.6113586, -104.8951111, 105.3670883
2: -72.7158813, 46.1605721, -76.8740845, 48.8779106, -121.5937958, 123.0346527
3: -77.1514893, 39.8565636, -81.6379242, 42.2296867, -119.3811569, 121.4944839
4: -72.1226959, 53.4341507, -76.2121964, 56.5844994, -128.7071686, 129.6463470
5: -64.0869293, 49.7719688, -67.7469559, 52.5961151, -116.6830444, 117.5189209
6: -59.2973900, 58.0655937, -62.7581062, 61.4671059, -120.7644882, 120.8237000
7: -65.1676102, 57.9393730, -68.9854279, 61.1216583, -126.2892685, 126.9248047
8: -84.0978088, 49.9053078, -88.6048813, 53.0617142, -137.1595154, 138.5101929
9: -58.7354584, 58.7163086, -62.2261391, 62.1867828, -120.9222412, 120.9424362

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9016685, upper bound: 132.9043877
time: 9.60 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9108734, upper bound: 132.9102413
time: 7.77 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -69.5974808, 55.6752396, -68.1397476, 54.5593185, -124.1567841, 123.8149796
1: -60.0971375, 48.9616241, -58.7186203, 47.9831886, -108.0803223, 107.6802444
2: -77.6256561, 49.2082863, -75.8996582, 48.2428665, -125.8685226, 125.1079407
3: -82.4195938, 42.4158249, -80.6034393, 41.6835518, -124.1031494, 123.0192566
4: -77.0379486, 56.9671783, -75.2607651, 55.8540039, -132.8919525, 132.2279358
5: -68.3729553, 53.0688896, -66.8936234, 51.9387741, -120.3117218, 119.9625092
6: -63.2544403, 61.9612122, -61.9505310, 60.6769180, -123.9313583, 123.9117355
7: -69.5783997, 61.8703766, -68.0984116, 60.3784065, -129.9567871, 129.9687805
8: -89.8109360, 52.9954147, -87.5404205, 52.3334312, -142.1443634, 140.5358124
9: -62.6431046, 62.6076050, -61.4197197, 61.3839035, -124.0270081, 124.0273209

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9019507, upper bound: 132.9050436
time: 8.75 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9114574, upper bound: 132.9114574
time: 6.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.80 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.80
Output dim: 8, lower bound: -132.9016685, upper bound: 132.9043877
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.80
Output dim: 8, lower bound: -132.9108734, upper bound: 132.9102413
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.80
Output dim: 8, lower bound: -132.9019507, upper bound: 132.9050436
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 16.80
Output dim: 8, lower bound: -132.9114574, upper bound: 132.9114574

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -59.2597389, 47.4436722, -54.9627380, 43.9611053, -103.2208405, 102.4064102
1: -51.2276573, 41.6323929, -47.5810394, 38.5711021, -89.7987595, 89.2134323
2: -66.0641937, 41.9002151, -61.2360878, 38.8478775, -104.9120636, 103.1362991
3: -69.9957962, 36.1245728, -64.8253860, 33.4347420, -103.4305420, 100.9499512
4: -65.6650162, 48.4822807, -61.0488663, 44.9275818, -110.5925980, 109.5311432
5: -58.2811241, 45.2819672, -54.0889320, 42.0413094, -100.3224335, 99.3708954
6: -53.7830391, 52.7126160, -49.7932625, 48.8782883, -102.6613312, 102.5058746
7: -59.1447792, 52.8670006, -54.8256187, 49.1980553, -108.3428268, 107.6926193
8: -76.8250122, 44.9915924, -71.5122452, 41.5071030, -118.3321075, 116.5038376
9: -53.2996025, 53.3581238, -49.4387589, 49.5779953, -102.8775940, 102.7968826

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8721891, upper bound: 132.8764752
time: 8.01 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8711881, upper bound: 132.8748231
time: 10.67 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -62.0882339, 49.7279282, -63.8132591, 51.0824471, -113.1706848, 113.5411835
1: -53.6183167, 43.6570816, -55.1213036, 44.8760948, -98.4944153, 98.7783813
2: -69.2135086, 43.9202919, -71.1196136, 45.1400528, -114.3535614, 115.0399017
3: -73.3948517, 37.9038162, -75.4497910, 38.9446182, -112.3394699, 113.3536072
4: -68.7095413, 50.8401985, -70.6621857, 52.2756500, -120.9851761, 121.5023804
5: -61.0305634, 47.4096870, -62.7231674, 48.7110977, -109.7416611, 110.1328583
6: -56.4017220, 55.2460365, -57.9719048, 56.8009872, -113.2027054, 113.2179337
7: -61.9967690, 55.2649612, -63.7502365, 56.7959099, -118.7926712, 119.0151978
8: -80.2609177, 47.3352470, -82.4320679, 48.6415367, -128.9024506, 129.7673187
9: -55.8747253, 55.8912277, -57.4756660, 57.4955368, -113.3702621, 113.3668823

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8853822, upper bound: 132.8869638
time: 8.52 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8841174, upper bound: 132.8839355
time: 9.28 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -63.6090240, 50.8721199, -54.1711845, 43.3295364, -106.9385529, 105.0433044
1: -55.0408363, 44.6892128, -46.9095726, 38.0082397, -93.0490646, 91.5987854
2: -70.9722519, 44.9450073, -60.3640327, 38.2808990, -109.2531509, 105.3090363
3: -75.2624130, 38.6813889, -63.9015808, 32.9423943, -108.2048035, 102.5829697
4: -70.5775833, 52.0158043, -60.1940231, 44.2702599, -114.8478394, 112.2098236
5: -62.5707664, 48.5748329, -53.3239250, 41.4505730, -104.0213394, 101.8987503
6: -57.7428589, 56.6050224, -49.0669861, 48.1710739, -105.9139252, 105.6720047
7: -63.5554085, 56.8007736, -54.0312996, 48.5321999, -112.0876083, 110.8320618
8: -82.5393906, 48.0676765, -70.5562668, 40.8540878, -123.3934784, 118.6239471
9: -57.2019157, 57.2387924, -48.7171860, 48.8584251, -106.0603409, 105.9559784

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8724564, upper bound: 132.8771861
time: 8.53 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8714900, upper bound: 132.8754492
time: 8.98 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -66.3690262, 53.0994835, -62.9166107, 50.3679428, -116.7369614, 116.0160828
1: -57.3668900, 46.6632996, -54.3641777, 44.2393684, -101.6062622, 101.0274811
2: -74.0408554, 46.9161339, -70.1322327, 44.4989128, -118.5397644, 117.0483704
3: -78.5767593, 40.4152756, -74.3980560, 38.3898468, -116.9665985, 114.8133240
4: -73.5480728, 54.3096619, -69.6943817, 51.5364647, -125.0845184, 124.0040359
5: -65.2471542, 50.6528435, -61.8568802, 48.0442924, -113.2914429, 112.5096970
6: -60.2944946, 59.0742340, -57.1525803, 56.0003166, -116.2948151, 116.2268143
7: -66.3349152, 59.1352959, -62.8499451, 56.0437660, -122.3786774, 121.9852295
8: -85.8883209, 50.3634758, -81.3540039, 47.9016991, -133.7900085, 131.7174835
9: -59.7159348, 59.7116547, -56.6584244, 56.6821747, -116.3980942, 116.3700714

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8856630, upper bound: 132.8879025
time: 9.81 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8843703, upper bound: 132.8843703
time: 8.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.09 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.09
Output dim: 8, lower bound: -132.8721891, upper bound: 132.8764752
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.09
Output dim: 8, lower bound: -132.8711881, upper bound: 132.8748231
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.09
Output dim: 8, lower bound: -132.8853822, upper bound: 132.8869638
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.09
Output dim: 8, lower bound: -132.8841174, upper bound: 132.8839355
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.09
Output dim: 8, lower bound: -132.8724564, upper bound: 132.8771861
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.09
Output dim: 8, lower bound: -132.8714900, upper bound: 132.8754492
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.09
Output dim: 8, lower bound: -132.8856630, upper bound: 132.8879025
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.09
Output dim: 8, lower bound: -132.8843703, upper bound: 132.8843703

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -52.1780815, 41.7723656, -53.8028946, 43.0338669, -95.2119446, 95.5752563
1: -45.2119370, 36.6213417, -46.5962753, 37.7481575, -82.9600906, 83.2175903
2: -58.1716614, 36.8506012, -59.9415665, 38.0237885, -96.1954269, 96.7921677
3: -61.6012115, 31.7363720, -63.4542198, 32.7108383, -94.3120422, 95.1905899
4: -57.9784431, 42.5936852, -59.7862053, 43.9635162, -101.9419556, 102.3798904
5: -51.3699760, 39.9773445, -52.9582367, 41.1736908, -92.5436707, 92.9355698
6: -47.2449646, 46.3942566, -48.7228851, 47.8409309, -95.0858917, 95.1171265
7: -51.9941711, 46.8452950, -53.6552505, 48.2111168, -100.2052917, 100.5005341
8: -68.2315521, 39.1678352, -70.1059418, 40.5529099, -108.7844543, 109.2737732
9: -46.8370323, 47.0030785, -48.3807220, 48.5345116, -95.3715210, 95.3837967

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8720319, upper bound: 132.8764752
time: 10.23 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8720319, upper bound: 132.8764752
time: 8.75 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -48.9026718, 39.0901871, -47.7294273, 38.1887932, -87.0914536, 86.8196106
1: -42.6439972, 34.2893867, -41.4658585, 33.4654999, -76.1094971, 75.7552414
2: -54.6191406, 34.4406700, -53.1865234, 33.7177887, -88.3369217, 87.6271973
3: -57.8079453, 29.5464497, -56.2816925, 28.9371700, -86.7451172, 85.8281403
4: -54.6496735, 39.7831001, -53.1906128, 38.9195862, -93.5692596, 92.9737091
5: -48.2263794, 37.5737152, -47.0556717, 36.6340332, -84.8604050, 84.6293869
6: -44.2136002, 43.5238075, -43.1551132, 42.4336929, -86.6472778, 86.6789246
7: -48.7580109, 44.3363152, -47.5585022, 43.0527649, -91.8107758, 91.8947906
8: -64.7336578, 35.9099007, -62.7158432, 35.5622139, -100.2958527, 98.6257324
9: -43.7622261, 43.9789429, -42.8406677, 43.0534935, -86.8157196, 86.8195953

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8710348, upper bound: 132.8748231
time: 9.87 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8710348, upper bound: 132.8748231
time: 8.78 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -54.7405624, 43.8394165, -62.5286446, 50.0536270, -104.7941818, 106.3680573
1: -47.3761215, 38.4464798, -54.0332718, 43.9648361, -91.3409576, 92.4797516
2: -61.0231895, 38.6640778, -69.6879883, 44.2219124, -105.2451019, 108.3520660
3: -64.6733551, 33.3559074, -73.9259186, 38.1507378, -102.8240891, 107.2818298
4: -60.7283974, 44.7367706, -69.2645645, 51.2128181, -111.9412003, 114.0013351
5: -53.8605461, 41.8984756, -61.4726067, 47.7479973, -101.6085434, 103.3710785
6: -49.6126060, 48.6828346, -56.7863922, 55.6532516, -105.2658539, 105.4692230
7: -54.5667000, 49.0122147, -62.4529877, 55.7041931, -110.2708893, 111.4652023
8: -71.3498001, 41.2837563, -80.8770294, 47.5796814, -118.9294815, 122.1607819
9: -49.1614532, 49.2936668, -56.3017464, 56.3409653, -105.5024185, 105.5954132

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8850843, upper bound: 132.8869638
time: 10.45 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8850843, upper bound: 132.8869638
time: 10.13 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -51.3160439, 41.0386543, -55.9170837, 44.7548447, -96.0708618, 96.9557343
1: -44.6764717, 35.9982796, -48.4321136, 39.2633972, -83.9398651, 84.4303894
2: -57.2977982, 36.1526871, -62.3194656, 39.4915428, -96.7893372, 98.4721527
3: -60.6944160, 31.0989494, -66.0871124, 34.0698929, -94.7643127, 97.1860657
4: -57.2413750, 41.8031807, -62.0793114, 45.7207642, -102.9621429, 103.8824921
5: -50.5680008, 39.3887901, -55.0246658, 42.7905121, -93.3585129, 94.4134521
6: -46.4516525, 45.6767502, -50.6977386, 49.7437897, -96.1954346, 96.3744888
7: -51.1741943, 46.3728104, -55.7822266, 50.0736885, -101.2478714, 102.1550369
8: -67.6727371, 37.9128571, -72.8432236, 42.1431122, -109.8158493, 110.7560806
9: -45.9394875, 46.1420784, -50.2430038, 50.3688545, -96.3083267, 96.3850708

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8838804, upper bound: 132.8839332
time: 7.10 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8838804, upper bound: 132.8839355
time: 8.31 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -56.3749695, 45.0777359, -53.0242348, 42.4148941, -98.7898636, 98.1019669
1: -48.8971405, 39.5607452, -45.9367790, 37.1961441, -86.0932846, 85.4975281
2: -62.9119072, 39.7730560, -59.0838928, 37.4672928, -100.3791962, 98.8569489
3: -66.6849518, 34.1987457, -62.5475845, 32.2260284, -98.9109802, 96.7463303
4: -62.7223015, 45.9988899, -58.9452667, 43.3172607, -106.0395584, 104.9441528
5: -55.5095978, 43.1551437, -52.2075348, 40.5925827, -96.1021805, 95.3626709
6: -51.0556564, 50.1429405, -48.0096893, 47.1465073, -98.2021637, 98.1526337
7: -56.2436829, 50.6495552, -52.8755112, 47.5566177, -103.8003006, 103.5250626
8: -73.7733688, 42.1030540, -69.1649551, 39.9114990, -113.6848602, 111.2680054
9: -50.5902786, 50.7419968, -47.6707420, 47.8282356, -98.4185104, 98.4127197

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8720829, upper bound: 132.8770166
time: 13.90 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8720829, upper bound: 132.8771861
time: 12.86 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -52.8217850, 42.1806755, -47.0426559, 37.6412544, -90.4630203, 89.2233276
1: -46.0865097, 37.0332489, -40.8864059, 32.9862518, -79.0727386, 77.9196548
2: -59.0440331, 37.1732941, -52.4355621, 33.2279663, -92.2719879, 89.6088562
3: -62.5562820, 31.8538170, -55.4838257, 28.5094948, -91.0657654, 87.3376465
4: -59.0997391, 42.9568214, -52.4503479, 38.3476868, -97.4474182, 95.4071655
5: -52.0917816, 40.5433350, -46.3939896, 36.1228142, -88.2145996, 86.9373245
6: -47.7689209, 47.0220146, -42.5292320, 41.8248024, -89.5937195, 89.5512238
7: -52.7299690, 47.9018936, -46.8705444, 42.4786148, -95.2085876, 94.7724380
8: -69.9206772, 38.6352844, -61.8855515, 34.9951286, -104.9157944, 100.5208359
9: -47.2717514, 47.4727478, -42.2192154, 42.4312286, -89.7029724, 89.6919632

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8710990, upper bound: 132.8752460
time: 8.80 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8710990, upper bound: 132.8754492
time: 8.60 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -58.9057884, 47.1186142, -61.6435928, 49.3466797, -108.2524719, 108.7622070
1: -51.0339622, 41.3661003, -53.2829437, 43.3343353, -94.3682938, 94.6490479
2: -65.7260742, 41.5746422, -68.7127838, 43.5872116, -109.3132858, 110.2874298
3: -69.7158127, 35.7969666, -72.8841019, 37.6013451, -107.3171539, 108.6810532
4: -65.4374924, 48.1125565, -68.3097229, 50.4783859, -115.9158783, 116.4222794
5: -57.9697342, 45.0571289, -60.6143188, 47.0887146, -105.0584335, 105.6714478
6: -53.3963814, 52.4060402, -55.9745712, 54.8617935, -108.2581787, 108.3806152
7: -58.7919884, 52.7886276, -61.5626793, 54.9594421, -113.7514343, 114.3513031
8: -76.8505936, 44.2010880, -79.8114777, 46.8498344, -123.7004242, 124.0125351
9: -52.8909836, 53.0061378, -55.4933052, 55.5374908, -108.4284744, 108.4994278

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8850746, upper bound: 132.8875144
time: 8.73 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8850746, upper bound: 132.8879025
time: 10.78 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -55.1851501, 44.0875511, -55.0918808, 44.0987816, -99.2839203, 99.1794281
1: -48.0735970, 38.7056923, -47.7341728, 38.6813965, -86.7549896, 86.4398499
2: -61.6652069, 38.8466988, -61.4141884, 38.9019203, -100.5671234, 100.2608795
3: -65.3835602, 33.3712502, -65.1209564, 33.5594139, -98.9429779, 98.4922028
4: -61.6304779, 44.9346962, -61.1947403, 45.0360336, -106.6665039, 106.1294403
5: -54.3845940, 42.3166809, -54.2263374, 42.1771240, -96.5617218, 96.5430069
6: -49.9610291, 49.1287079, -49.9433289, 49.0102196, -98.9712524, 99.0720367
7: -55.0942917, 49.8913193, -54.9573212, 49.3829727, -104.4772415, 104.8486404
8: -72.7896271, 40.6062012, -71.8502960, 41.4641685, -114.2537994, 112.4564972
9: -49.4076118, 49.5880165, -49.4938393, 49.6232834, -99.0308990, 99.0818558

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8838382, upper bound: 132.8841173
time: 7.55 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8838642, upper bound: 132.8843703
time: 8.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 17.49 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -132.8720319, upper bound: 132.8764752
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -132.8720319, upper bound: 132.8764752
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -132.8710348, upper bound: 132.8748231
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -132.8710348, upper bound: 132.8748231
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -132.8850843, upper bound: 132.8869638
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -132.8850843, upper bound: 132.8869638
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -132.8838804, upper bound: 132.8839332
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -132.8838804, upper bound: 132.8839355
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -132.8720829, upper bound: 132.8770166
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -132.8720829, upper bound: 132.8771861
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -132.8710990, upper bound: 132.8752460
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -132.8710990, upper bound: 132.8754492
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -132.8850746, upper bound: 132.8875144
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -132.8850746, upper bound: 132.8879025
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -132.8838382, upper bound: 132.8841173
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.49
Output dim: 8, lower bound: -132.8838642, upper bound: 132.8843703

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -52.1780815, 41.7723656, -50.5336685, 40.4282837, -92.6063690, 92.3060303
1: -45.2119370, 36.6213417, -43.8482819, 35.4198341, -80.6317749, 80.4696045
2: -58.1716614, 36.8506012, -56.3614006, 35.7028122, -93.8744583, 93.2119904
3: -61.6012115, 31.7363720, -59.5837975, 30.6407166, -92.2419281, 91.3201675
4: -57.9784431, 42.5936852, -56.2463531, 41.2283554, -99.2068024, 98.8400421
5: -51.3699760, 39.9773445, -49.7950134, 38.7358551, -90.1058350, 89.7723465
6: -47.2449646, 46.3942566, -45.7293358, 44.9229355, -92.1679001, 92.1235886
7: -51.9941711, 46.8452950, -50.3704071, 45.4715385, -97.4656982, 97.2156982
8: -68.2315521, 39.1678352, -66.2148209, 37.8267555, -106.0583038, 105.3826599
9: -46.8370323, 47.0030785, -45.3686600, 45.5531464, -92.3901596, 92.3717346

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8452065, upper bound: 132.8515185
time: 9.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8421387, upper bound: 132.8446923
time: 9.97 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -52.1780815, 41.7723656, -54.6256943, 43.6531715, -95.8312531, 96.3980560
1: -45.2119370, 36.6213417, -47.4374542, 38.2883797, -83.5003204, 84.0587845
2: -58.1716614, 36.8506012, -60.9792252, 38.5523529, -96.7240067, 97.8298187
3: -61.6012115, 31.7363720, -64.5392685, 33.0537987, -94.6550140, 96.2756348
4: -57.9784431, 42.5936852, -60.8760757, 44.5465240, -102.5249634, 103.4697571
5: -51.3699760, 39.9773445, -53.8273621, 41.8352737, -93.2052460, 93.8047028
6: -47.2449646, 46.3942566, -49.4419327, 48.5781860, -95.8231506, 95.8361740
7: -51.9941711, 46.8452950, -54.5140915, 49.1748505, -101.1690216, 101.3593903
8: -68.2315521, 39.1678352, -71.6053085, 40.6982346, -108.9297714, 110.7731476
9: -46.8370323, 47.0030785, -49.0308914, 49.1977158, -96.0347290, 96.0339584

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8452065, upper bound: 132.8516573
time: 11.25 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8421387, upper bound: 132.8447202
time: 10.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -48.9026718, 39.0901871, -44.7696533, 35.8224640, -84.7251282, 83.8598328
1: -42.6439972, 34.2893867, -38.9861221, 31.3799019, -74.0238953, 73.2754974
2: -54.6191406, 34.4406700, -49.9565201, 31.6191101, -86.2382507, 84.3971863
3: -57.8079453, 29.5464497, -52.7810059, 27.0535698, -84.8615112, 82.3274536
4: -54.6496735, 39.7831001, -49.9856453, 36.4378624, -91.0875244, 89.7687378
5: -48.2263794, 37.5737152, -44.1844139, 34.4293556, -82.6557236, 81.7581329
6: -44.2136002, 43.5238075, -40.4515190, 39.8017426, -84.0153351, 83.9753265
7: -48.7580109, 44.3363152, -44.5812645, 40.5764961, -89.3345032, 88.9175720
8: -64.7336578, 35.9099007, -59.1893463, 33.0840683, -97.8177261, 95.0992355
9: -43.7622261, 43.9789429, -40.1237640, 40.3485069, -84.1107330, 84.1026917

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8437926, upper bound: 132.8495369
time: 9.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8413287, upper bound: 132.8431906
time: 7.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -48.9026718, 39.0901871, -48.7339935, 38.9443321, -87.8470001, 87.8241653
1: -42.6439972, 34.2893867, -42.4655724, 34.1555977, -76.7995911, 76.7549438
2: -54.6191406, 34.4406700, -54.4374161, 34.3789978, -88.9981308, 88.8780594
3: -57.8079453, 29.5464497, -57.5775604, 29.3969364, -87.2048721, 87.1240082
4: -54.6496735, 39.7831001, -54.4860802, 39.6521759, -94.3018494, 94.2691727
5: -48.2263794, 37.5737152, -48.0954323, 37.4345856, -85.6609650, 85.6691437
6: -44.2136002, 43.5238075, -44.0436058, 43.3409882, -87.5545807, 87.5674133
7: -48.7580109, 44.3363152, -48.6048393, 44.1781845, -92.9361954, 92.9411469
8: -64.7336578, 35.9099007, -64.4338989, 35.8585510, -100.5922089, 100.3437958
9: -43.7622261, 43.9789429, -43.6727066, 43.8855934, -87.6478195, 87.6516418

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8437926, upper bound: 132.8497663
time: 9.48 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8413287, upper bound: 132.8432638
time: 8.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -54.7405624, 43.8394165, -58.9459457, 47.1881943, -101.9287491, 102.7853622
1: -47.3761215, 38.4464798, -51.0103645, 41.3950233, -88.7711487, 89.4568405
2: -61.0231895, 38.6640778, -65.7471619, 41.6506805, -102.6738663, 104.4112396
3: -64.6733551, 33.3559074, -69.6628113, 35.8986816, -100.5720139, 103.0187225
4: -60.7283974, 44.7367706, -65.3791885, 48.2128944, -108.9412842, 110.1159592
5: -53.8605461, 41.8984756, -57.9899597, 45.0684509, -98.9290009, 99.8884354
6: -49.6126060, 48.6828346, -53.4909859, 52.4337196, -102.0463104, 102.1738205
7: -54.5667000, 49.0122147, -58.8280869, 52.6838875, -107.2505875, 107.8402863
8: -71.3498001, 41.2837563, -76.6041870, 44.5861244, -115.9359055, 117.8879395
9: -49.1614532, 49.2936668, -52.9893417, 53.0600739, -102.2215271, 102.2830048

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8513347, upper bound: 132.8571988
time: 11.49 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8478748, upper bound: 132.8489409
time: 8.52 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -54.7405624, 43.8394165, -62.9821625, 50.3686752, -105.1092377, 106.8215637
1: -47.3761215, 38.4464798, -54.5577202, 44.2299385, -91.6060486, 93.0041962
2: -61.0231895, 38.6640778, -70.3058777, 44.4769745, -105.5001602, 108.9699554
3: -64.6733551, 33.3559074, -74.5547333, 38.2621841, -102.9355392, 107.9106369
4: -60.7283974, 44.7367706, -69.9523315, 51.4910583, -112.2194290, 114.6891022
5: -53.8605461, 41.8984756, -61.9751472, 48.1269989, -101.9875488, 103.8736191
6: -49.6126060, 48.6828346, -57.1646004, 56.0483284, -105.6609344, 105.8474350
7: -54.5667000, 49.0122147, -62.9242897, 56.3478241, -110.9145203, 111.9364777
8: -71.3498001, 41.2837563, -81.9324493, 47.4215088, -118.7713013, 123.2162018
9: -49.1614532, 49.2936668, -56.6103745, 56.6636200, -105.8250580, 105.9040375

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8513347, upper bound: 132.8577114
time: 10.04 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8478748, upper bound: 132.8490791
time: 11.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -51.3160439, 41.0386543, -52.6031303, 42.1125870, -93.4286118, 93.6417694
1: -44.6764717, 35.9982796, -45.6482582, 36.9167862, -81.5932465, 81.6465302
2: -57.2977982, 36.1526871, -58.7019653, 37.1375275, -94.4353256, 94.8546371
3: -60.6944160, 31.0989494, -62.1590652, 31.9867992, -92.6812057, 93.2580109
4: -57.2413750, 41.8031807, -58.5151672, 42.9522095, -100.1935730, 100.3183441
5: -50.5680008, 39.3887901, -51.8077927, 40.3296127, -90.8976135, 91.1965790
6: -46.4516525, 45.6767502, -47.6672401, 46.7943840, -93.2460327, 93.3439941
7: -51.1741943, 46.3728104, -52.4621468, 47.3029442, -98.4771271, 98.8349609
8: -67.6727371, 37.9128571, -68.9071350, 39.3796234, -107.0523529, 106.8199921
9: -45.9394875, 46.1420784, -47.1976929, 47.3541222, -93.2935867, 93.3397675

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8498910, upper bound: 132.8544951
time: 7.19 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8469510, upper bound: 132.8469510
time: 8.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -51.3160439, 41.0386543, -56.5177994, 45.1990585, -96.5150833, 97.5564423
1: -44.6764717, 35.9982796, -49.0907822, 39.6547699, -84.3312302, 85.0890579
2: -57.2977982, 36.1526871, -63.1209335, 39.8677292, -97.1655273, 99.2736130
3: -60.6944160, 31.0989494, -66.9021912, 34.2827072, -94.9771042, 98.0011444
4: -57.2413750, 41.8031807, -62.9513168, 46.1253510, -103.3667297, 104.7545013
5: -50.5680008, 39.3887901, -55.6718407, 43.2948074, -93.8628082, 95.0606308
6: -46.4516525, 45.6767502, -51.2207565, 50.2911148, -96.7427673, 96.8975067
7: -51.1741943, 46.3728104, -56.4280777, 50.8612328, -102.0354080, 102.8008881
8: -67.6727371, 37.9128571, -74.0859604, 42.1113358, -109.7840729, 111.9988098
9: -45.9394875, 46.1420784, -50.7043991, 50.8437576, -96.7832260, 96.8464813

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 175

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8498910, upper bound: 132.8550328
time: 11.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8469510, upper bound: 132.8473123
time: 9.32 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -56.3749695, 45.0777359, -50.5336685, 40.4282837, -96.8032532, 95.6114044
1: -48.8971405, 39.5607452, -43.8482819, 35.4198341, -84.3169708, 83.4090271
2: -62.9119072, 39.7730560, -56.3614006, 35.7028122, -98.6147156, 96.1344604
3: -66.6849518, 34.1987457, -59.5837975, 30.6407166, -97.3256683, 93.7825470
4: -62.7223015, 45.9988899, -56.2463531, 41.2283554, -103.9506531, 102.2452393
5: -55.5095978, 43.1551437, -49.7950134, 38.7358551, -94.2454529, 92.9501495
6: -51.0556564, 50.1429405, -45.7293358, 44.9229355, -95.9785919, 95.8722763
7: -56.2436829, 50.6495552, -50.3704071, 45.4715385, -101.7152023, 101.0199585
8: -73.7733688, 42.1030540, -66.2148209, 37.8267555, -111.6001129, 108.3178711
9: -50.5902786, 50.7419968, -45.3686600, 45.5531464, -96.1434174, 96.1106415

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8452391, upper bound: 132.8520239
time: 9.49 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8424440, upper bound: 132.8464102
time: 7.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -56.3749695, 45.0777359, -54.6254005, 43.6528778, -100.0278473, 99.7031250
1: -48.8971405, 39.5607452, -47.4371681, 38.2879524, -87.1850891, 86.9979095
2: -62.9119072, 39.7730560, -60.9787140, 38.5520782, -101.4639893, 100.7517700
3: -66.6849518, 34.1987457, -64.5389481, 33.0534515, -99.7383957, 98.7376938
4: -62.7223015, 45.9988899, -60.8757286, 44.5462112, -107.2685089, 106.8746185
5: -55.5095978, 43.1551437, -53.8269920, 41.8348923, -97.3444901, 96.9821167
6: -51.0556564, 50.1429405, -49.4415703, 48.5778427, -99.6334991, 99.5845108
7: -56.2436829, 50.6495552, -54.5136032, 49.1745186, -105.4181976, 105.1631546
8: -73.7733688, 42.1030540, -71.6047592, 40.6980057, -114.4713669, 113.7078094
9: -50.5902786, 50.7419968, -49.0305634, 49.1973000, -99.7875671, 99.7725449

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8452391, upper bound: 132.8531237
time: 11.27 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8424440, upper bound: 132.8464102
time: 10.09 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -52.8217850, 42.1806755, -44.7696533, 35.8224640, -88.6442261, 86.9503326
1: -46.0865097, 37.0332489, -38.9861221, 31.3799019, -77.4664001, 76.0193710
2: -59.0440331, 37.1732941, -49.9565201, 31.6191101, -90.6631470, 87.1298141
3: -62.5562820, 31.8538170, -52.7810059, 27.0535698, -89.6098480, 84.6348267
4: -59.0997391, 42.9568214, -49.9856453, 36.4378624, -95.5375824, 92.9424591
5: -52.0917816, 40.5433350, -44.1844139, 34.4293556, -86.5211334, 84.7277451
6: -47.7689209, 47.0220146, -40.4515190, 39.8017426, -87.5706635, 87.4735107
7: -52.7299690, 47.9018936, -44.5812645, 40.5764961, -93.3064575, 92.4831543
8: -69.9206772, 38.6352844, -59.1893463, 33.0840683, -103.0047379, 97.8246307
9: -47.2717514, 47.4727478, -40.1237640, 40.3485069, -87.6202545, 87.5965118

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 175

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8438189, upper bound: 132.8499264
time: 9.47 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8415031, upper bound: 132.8444495
time: 8.31 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -52.8217850, 42.1806755, -48.7334518, 38.9438820, -91.7656403, 90.9141235
1: -46.0865097, 37.0332489, -42.4650917, 34.1551590, -80.2416458, 79.4983368
2: -59.0440331, 37.1732941, -54.4368324, 34.3786621, -93.4226990, 91.6101227
3: -62.5562820, 31.8538170, -57.5767670, 29.3966312, -91.9529037, 89.4305878
4: -59.0997391, 42.9568214, -54.4856415, 39.6516037, -98.7513428, 97.4424362
5: -52.0917816, 40.5433350, -48.0949516, 37.4341850, -89.5259705, 88.6382828
6: -47.7689209, 47.0220146, -44.0430183, 43.3405113, -91.1094360, 91.0650330
7: -52.7299690, 47.9018936, -48.6042480, 44.1777840, -96.9077530, 96.5061417
8: -69.9206772, 38.6352844, -64.4332657, 35.8579941, -105.7786713, 103.0685501
9: -47.2717514, 47.4727478, -43.6722298, 43.8851242, -91.1568680, 91.1449738

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 175

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8438189, upper bound: 132.8511030
time: 9.89 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8415031, upper bound: 132.8455034
time: 7.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -58.9057884, 47.1186142, -58.9459457, 47.1881943, -106.0939789, 106.0645599
1: -51.0339622, 41.3661003, -51.0103645, 41.3950233, -92.4289856, 92.3764648
2: -65.7260742, 41.5746422, -65.7471619, 41.6506805, -107.3767548, 107.3218079
3: -69.7158127, 35.7969666, -69.6628113, 35.8986816, -105.6144867, 105.4597778
4: -65.4374924, 48.1125565, -65.3791885, 48.2128944, -113.6503906, 113.4917450
5: -57.9697342, 45.0571289, -57.9899597, 45.0684509, -103.0381851, 103.0470886
6: -53.3963814, 52.4060402, -53.4909859, 52.4337196, -105.8300781, 105.8970261
7: -58.7919884, 52.7886276, -58.8280869, 52.6838875, -111.4758759, 111.6167145
8: -76.8505936, 44.2010880, -76.6041870, 44.5861244, -121.4366989, 120.8052597
9: -52.8909836, 53.0061378, -52.9893417, 53.0600739, -105.9510574, 105.9954681

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8513853, upper bound: 132.8579575
time: 11.53 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8483660, upper bound: 132.8511722
time: 10.27 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -58.9057884, 47.1186142, -62.9821625, 50.3686752, -109.2744598, 110.1007767
1: -51.0339622, 41.3661003, -54.5577202, 44.2299385, -95.2638779, 95.9238205
2: -65.7260742, 41.5746422, -70.3058777, 44.4769745, -110.2030487, 111.8805084
3: -69.7158127, 35.7969666, -74.5547333, 38.2621841, -107.9779968, 110.3516922
4: -65.4374924, 48.1125565, -69.9523315, 51.4910583, -116.9285355, 118.0648880
5: -57.9697342, 45.0571289, -61.9751472, 48.1269989, -106.0967255, 107.0322723
6: -53.3963814, 52.4060402, -57.1646004, 56.0483284, -109.4447098, 109.5706253
7: -58.7919884, 52.7886276, -62.9242897, 56.3478241, -115.1398163, 115.7129059
8: -76.8505936, 44.2010880, -81.9324493, 47.4215088, -124.2720947, 126.1334991
9: -52.8909836, 53.0061378, -56.6103745, 56.6636200, -109.5545959, 109.6165009

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8513853, upper bound: 132.8601539
time: 8.91 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8483660, upper bound: 132.8532035
time: 9.51 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -55.1851501, 44.0875511, -52.6031303, 42.1125870, -97.2977371, 96.6906662
1: -48.0735970, 38.7056923, -45.6482582, 36.9167862, -84.9903870, 84.3539429
2: -61.6652069, 38.8466988, -58.7019653, 37.1375275, -98.8027344, 97.5486526
3: -65.3835602, 33.3712502, -62.1590652, 31.9867992, -97.3703613, 95.5303192
4: -61.6304779, 44.9346962, -58.5151672, 42.9522095, -104.5826874, 103.4498596
5: -54.3845940, 42.3166809, -51.8077927, 40.3296127, -94.7142029, 94.1244736
6: -49.9610291, 49.1287079, -47.6672401, 46.7943840, -96.7554169, 96.7959442
7: -55.0942917, 49.8913193, -52.4621468, 47.3029442, -102.3972168, 102.3534698
8: -72.7896271, 40.6062012, -68.9071350, 39.3796234, -112.1692505, 109.5133362
9: -49.4076118, 49.5880165, -47.1976929, 47.3541222, -96.7617340, 96.7857056

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8498926, upper bound: 132.8550419
time: 14.00 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8473123, upper bound: 132.8484555
time: 10.63 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.79 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8452065, upper bound: 132.8515185
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8421387, upper bound: 132.8446923
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8452065, upper bound: 132.8516573
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8421387, upper bound: 132.8447202
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8437926, upper bound: 132.8495369
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8413287, upper bound: 132.8431906
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8437926, upper bound: 132.8497663
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8413287, upper bound: 132.8432638
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8513347, upper bound: 132.8571988
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8478748, upper bound: 132.8489409
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8513347, upper bound: 132.8577114
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8478748, upper bound: 132.8490791
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8498910, upper bound: 132.8544951
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8469510, upper bound: 132.8469510
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8498910, upper bound: 132.8550328
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8469510, upper bound: 132.8473123
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8452391, upper bound: 132.8520239
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8424440, upper bound: 132.8464102
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8452391, upper bound: 132.8531237
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8424440, upper bound: 132.8464102
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8438189, upper bound: 132.8499264
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8415031, upper bound: 132.8444495
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8438189, upper bound: 132.8511030
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8415031, upper bound: 132.8455034
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8513853, upper bound: 132.8579575
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8483660, upper bound: 132.8511722
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8513853, upper bound: 132.8601539
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8483660, upper bound: 132.8532035
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8498926, upper bound: 132.8550419
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.79
Output dim: 8, lower bound: -132.8473123, upper bound: 132.8484555
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.79
Output dim: 8, lower bound: -132.8838642, upper bound: 132.8843703
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=145.0349884033203
rel_dist={8: [-132.92639238641755, 132.92639237568028]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9192835, upper bound: 132.9189705
time: 13.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9201984, upper bound: 132.9201984
time: 10.64 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 24.27 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 24.27
Output dim: 8, lower bound: -132.9192835, upper bound: 132.9189705
IS_A2, status: Status.UNKNOWN, split count: 1, time: 24.27
Output dim: 8, lower bound: -132.9201984, upper bound: 132.9201984

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -65.2420654, 52.2499275, -66.7300186, 53.4355507, -118.6776123, 118.9799500
1: -56.2837639, 45.9009018, -57.5373764, 46.9672394, -103.2510071, 103.4382629
2: -72.7158813, 46.1605721, -74.3516541, 47.2298393, -119.9457245, 120.5122223
3: -77.1514893, 39.8565636, -78.9165192, 40.7902451, -117.9417191, 118.7730713
4: -72.1226959, 53.4341507, -73.7327194, 54.6741142, -126.7967911, 127.1668472
5: -64.0869293, 49.7719688, -65.5269928, 50.8836746, -114.9706039, 115.2989655
6: -59.2973900, 58.0655937, -60.6587830, 59.4050980, -118.7024841, 118.7243805
7: -65.1676102, 57.9393730, -66.6701050, 59.1915970, -124.3592072, 124.6094818
8: -84.0978088, 49.9053078, -85.8709183, 51.1483727, -135.2461853, 135.7762146
9: -58.7354584, 58.7163086, -60.1087494, 60.0825844, -118.8180389, 118.8250504

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9008736, upper bound: 132.9019850
time: 10.01 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9104399, upper bound: 132.9101629
time: 9.23 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -69.5974808, 55.6752396, -66.2527008, 53.0564499, -122.6538925, 121.9279404
1: -60.0971375, 48.9616241, -57.1265907, 46.6352882, -106.7324219, 106.0882111
2: -77.6256561, 49.2082863, -73.8219452, 46.8866005, -124.5122528, 123.0302200
3: -82.4195938, 42.4158249, -78.3760986, 40.5051727, -122.9247589, 120.7919235
4: -77.0379486, 56.9671783, -73.2247849, 54.2860031, -131.3239441, 130.1919556
5: -68.3729553, 53.0688896, -65.0677948, 50.5314941, -118.9044495, 118.1366882
6: -63.2544403, 61.9612122, -60.2251358, 58.9827957, -122.2372284, 122.1863480
7: -69.5783997, 61.8703766, -66.1969070, 58.7914009, -128.3697815, 128.0672913
8: -89.8109360, 52.9954147, -85.2827606, 50.7648048, -140.5757294, 138.2781677
9: -62.6431046, 62.6076050, -59.6858482, 59.6572571, -122.3003616, 122.2934570

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9013360, upper bound: 132.9025517
time: 14.33 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9008736, upper bound: 132.9114280
time: 11.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.47 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.47
Output dim: 8, lower bound: -132.9008736, upper bound: 132.9019850
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.47
Output dim: 8, lower bound: -132.9104399, upper bound: 132.9101629
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.47
Output dim: 8, lower bound: -132.9013360, upper bound: 132.9025517
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.47
Output dim: 8, lower bound: -132.9008736, upper bound: 132.9114280

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -54.5377808, 43.6422081, -52.8868942, 42.3086548, -96.8464279, 96.5290985
1: -47.2310638, 38.2603455, -45.8340225, 37.0842628, -84.3153229, 84.0943680
2: -60.8098907, 38.5314331, -58.9544601, 37.3670425, -98.1769257, 97.4858932
3: -64.3471222, 33.1642265, -62.3697166, 32.1214142, -96.4685135, 95.5339355
4: -60.5693436, 44.5548553, -58.7965393, 43.1914825, -103.7608185, 103.3513947
5: -53.6838570, 41.7355766, -52.0810318, 40.4913406, -94.1751862, 93.8166046
6: -49.4176407, 48.4887619, -47.8879623, 47.0159988, -96.4336395, 96.3767242
7: -54.3864632, 48.8594818, -52.7315140, 47.4529114, -101.8393631, 101.5909958
8: -71.0786896, 41.1139336, -69.0385284, 39.7761993, -110.8548889, 110.1524658
9: -49.0048599, 49.1306381, -47.5243263, 47.6794510, -96.6843033, 96.6549606

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8706058, upper bound: 132.8721726
time: 8.83 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8702813, upper bound: 132.8717058
time: 9.72 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -59.7608337, 47.8629875, -61.6143875, 49.3260574, -109.0868835, 109.4773483
1: -51.6462021, 41.9986763, -53.2704353, 43.2993240, -94.9455185, 95.2691116
2: -66.6279526, 42.2645378, -68.7027206, 43.5640030, -110.1919556, 110.9672546
3: -70.6142883, 36.4624519, -72.8336487, 37.5645370, -108.1788177, 109.2960968
4: -66.1885605, 48.9191704, -68.2770386, 50.4419632, -116.6305161, 117.1962128
5: -58.7662926, 45.6652374, -60.5904999, 47.0665398, -105.8328323, 106.2557220
6: -54.2568626, 53.1647377, -55.9541245, 54.8256645, -109.0825272, 109.1188660
7: -59.6538010, 53.2861557, -61.5278015, 54.9450722, -114.5988770, 114.8139572
8: -77.4259186, 45.4417458, -79.8120041, 46.8027077, -124.2286224, 125.2537537
9: -53.7602730, 53.8074036, -55.4437294, 55.4821587, -109.2424164, 109.2511292

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8843941, upper bound: 132.8850096
time: 9.06 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8839771, upper bound: 132.8838568
time: 8.09 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -58.8703613, 47.0589409, -52.4953766, 41.9982605, -100.8686218, 99.5543137
1: -51.0311584, 41.3064117, -45.4976044, 36.8150444, -87.8462067, 86.8040161
2: -65.7026596, 41.5633888, -58.5227509, 37.0863495, -102.7890091, 100.0861359
3: -69.5891266, 35.7157745, -61.9330750, 31.8880310, -101.4771576, 97.6488419
4: -65.4668808, 48.0738411, -58.3807755, 42.8724518, -108.3393326, 106.4546204
5: -57.9583893, 45.0194855, -51.7071648, 40.2001801, -98.1585693, 96.7266541
6: -53.3586006, 52.3665085, -47.5314293, 46.6734161, -100.0320129, 99.8979340
7: -58.7789307, 52.7806778, -52.3467484, 47.1252098, -105.9041290, 105.1274261
8: -76.7764664, 44.1713829, -68.5512085, 39.4630737, -116.2395401, 112.7225952
9: -52.8891525, 52.9952011, -47.1784515, 47.3333092, -100.2224579, 100.1736526

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8710224, upper bound: 132.8727201
time: 8.70 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8707488, upper bound: 132.8722352
time: 8.26 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -63.9751091, 51.1842537, -61.0838966, 48.9025421, -112.8776474, 112.2681351
1: -55.3440018, 44.9583549, -52.8152695, 42.9269485, -98.2709503, 97.7736206
2: -71.3830719, 45.2153015, -68.1149750, 43.1835289, -114.5666046, 113.3302689
3: -75.7191315, 38.9338150, -72.2256622, 37.2429657, -112.9620972, 111.1594772
4: -70.9558563, 52.3403549, -67.7109528, 50.0079575, -120.9638062, 120.0513077
5: -62.9230499, 48.8604355, -60.0777321, 46.6744156, -109.5974655, 108.9381714
6: -58.0928268, 56.9356346, -55.4689445, 54.3557663, -112.4485931, 112.4045792
7: -63.9275360, 57.1043549, -60.9994469, 54.4999695, -118.4274902, 118.1038055
8: -82.9765167, 48.4133148, -79.1612930, 46.3754349, -129.3519135, 127.5746078
9: -57.5425606, 57.5678902, -54.9710045, 55.0089951, -112.5515594, 112.5388947

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8847917, upper bound: 132.8857257
time: 9.76 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8843364, upper bound: 132.8843364
time: 9.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.48 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.48
Output dim: 8, lower bound: -132.8706058, upper bound: 132.8721726
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.48
Output dim: 8, lower bound: -132.8702813, upper bound: 132.8717058
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.48
Output dim: 8, lower bound: -132.8843941, upper bound: 132.8850096
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.48
Output dim: 8, lower bound: -132.8839771, upper bound: 132.8838568
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.48
Output dim: 8, lower bound: -132.8710224, upper bound: 132.8727201
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.48
Output dim: 8, lower bound: -132.8707488, upper bound: 132.8722352
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.48
Output dim: 8, lower bound: -132.8847917, upper bound: 132.8857257
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.48
Output dim: 8, lower bound: -132.8843364, upper bound: 132.8843364

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -48.0373878, 38.4441147, -48.5196533, 38.8181686, -86.8555450, 86.9637527
1: -41.7186623, 33.6922836, -42.1303635, 34.0124245, -75.7310867, 75.8226471
2: -53.5766258, 33.9300766, -54.0921822, 34.2749481, -87.8515625, 88.0222549
3: -56.6651077, 29.1189766, -57.2071228, 29.3957634, -86.0608673, 86.3260956
4: -53.5145950, 39.1550827, -54.0491257, 39.5588112, -93.0733948, 93.2042007
5: -47.3528976, 36.8775215, -47.8262100, 37.2260056, -84.5789032, 84.7037277
6: -43.4359245, 42.7085953, -43.8694153, 43.1315804, -86.5674973, 86.5780029
7: -47.8483124, 43.3424149, -48.3371887, 43.7427750, -91.5910873, 91.6796036
8: -63.1844788, 35.7749214, -63.7317314, 36.1837921, -99.3682632, 99.5066452
9: -43.0898590, 43.3083076, -43.5496674, 43.7608604, -86.8507233, 86.8579712

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8430142, upper bound: 132.8427922
time: 9.50 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8415182, upper bound: 132.8422310
time: 10.10 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -44.6749268, 35.7053375, -41.5013657, 33.2243271, -77.8992538, 77.2066879
1: -39.0546722, 31.3064880, -36.2228088, 29.1130886, -68.1677551, 67.5292892
2: -49.8996239, 31.4638157, -46.3134232, 29.3168831, -79.2164993, 77.7772369
3: -52.7700043, 26.8891792, -48.9469681, 25.0385933, -77.8085785, 75.8361511
4: -50.0535507, 36.2685013, -46.4506454, 33.7386360, -83.7921677, 82.7191391
5: -44.1138840, 34.3902092, -41.0013313, 31.9895325, -76.1033936, 75.3915405
6: -40.3367691, 39.7538490, -37.4675369, 36.9061317, -77.2429047, 77.2213745
7: -44.5004959, 40.7049713, -41.3071518, 37.7963791, -82.2968750, 82.0121155
8: -59.4826050, 32.5255432, -55.1771774, 30.4388790, -89.9214783, 87.7027130
9: -39.9481583, 40.1913910, -37.1749496, 37.4208069, -77.3689651, 77.3663177

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8427425, upper bound: 132.8423469
time: 11.54 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8412851, upper bound: 132.8417789
time: 8.22 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -52.5643997, 42.0965614, -56.5679932, 45.2752914, -97.8396912, 98.6645508
1: -45.5379448, 36.9053116, -48.9830475, 39.7176895, -85.2556305, 85.8883438
2: -58.6106834, 37.1257362, -63.0737114, 39.9469299, -98.5576172, 100.1994400
3: -62.0813141, 32.0018272, -66.8468475, 34.4341240, -96.5154266, 98.8486786
4: -58.3839035, 42.9352188, -62.7886047, 46.2413940, -104.6252899, 105.7238235
5: -51.7428436, 40.2701797, -55.6623611, 43.2784386, -95.0212860, 95.9325333
6: -47.6126137, 46.7464066, -51.2835121, 50.3147583, -97.9273682, 98.0299225
7: -52.3870468, 47.1689377, -56.4203758, 50.6480789, -103.0351257, 103.5893097
8: -68.6991425, 39.5141258, -73.6937027, 42.6354370, -111.3345718, 113.2078247
9: -47.1917725, 47.3489990, -50.8270683, 50.9455032, -98.1372604, 98.1760635

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8502072, upper bound: 132.8486593
time: 10.21 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8478611, upper bound: 132.8476810
time: 7.87 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -49.3870010, 39.4912796, -49.0120697, 39.2429619, -88.6299591, 88.5033493
1: -43.0546532, 34.6451492, -42.6132774, 34.4139900, -77.4686432, 77.2584229
2: -55.1668358, 34.7880020, -54.7011642, 34.5994263, -89.7662659, 89.4891663
3: -58.4073334, 29.8878880, -57.9282990, 29.7844658, -88.1917877, 87.8161850
4: -55.1615753, 40.2109146, -54.6238708, 39.9926109, -95.1541824, 94.8347855
5: -48.6937790, 37.9436264, -48.3152847, 37.6522789, -86.3460541, 86.2589111
6: -44.6784248, 43.9642105, -44.3802834, 43.6104279, -88.2888489, 88.3444901
7: -49.2479820, 44.7421608, -48.8663826, 44.2476044, -93.4955902, 93.6085434
8: -65.3202972, 36.3474579, -64.5128784, 36.4588852, -101.7791748, 100.8603287
9: -44.2050400, 44.4183960, -43.9442406, 44.1447601, -88.3498001, 88.3626251

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8499592, upper bound: 132.8479261
time: 9.05 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8475909, upper bound: 132.8470092
time: 9.06 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -52.1661987, 41.6993179, -48.1325836, 38.5101852, -90.6763687, 89.8318787
1: -45.3503990, 36.5804863, -41.7993584, 33.7469330, -79.0973358, 78.3798447
2: -58.2456398, 36.8075256, -53.6650543, 33.9948769, -92.2405167, 90.4725800
3: -61.6623650, 31.5481701, -56.7751541, 29.1645012, -90.8268661, 88.3233032
4: -58.2009277, 42.5007362, -53.6350746, 39.2433281, -97.4442444, 96.1358109
5: -51.4243279, 40.0094719, -47.4571114, 36.9372940, -88.3616180, 87.4665833
6: -47.1821671, 46.3988342, -43.5184746, 42.7908249, -89.9729919, 89.9173126
7: -52.0375824, 47.0949936, -47.9542885, 43.4172745, -95.4548569, 95.0492706
8: -68.6528473, 38.6553841, -63.2497444, 35.8715820, -104.5244141, 101.9051285
9: -46.7835922, 46.9868393, -43.2077103, 43.4166069, -90.2001877, 90.1945419

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8447966, upper bound: 132.8449252
time: 9.19 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8432023, upper bound: 132.8443572
time: 9.51 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -48.9596100, 39.0825539, -41.1691284, 32.9619331, -81.9215393, 80.2516785
1: -42.8364677, 34.3073807, -35.9341507, 28.8873177, -71.7237854, 70.2415314
2: -54.7609940, 34.4474182, -45.9435043, 29.0766335, -83.8376312, 80.3909225
3: -57.9728012, 29.3943558, -48.5802231, 24.8421593, -82.8149490, 77.9745789
4: -54.9407196, 39.7445068, -46.0960999, 33.4665947, -88.4073181, 85.8406067
5: -48.3498688, 37.6472664, -40.6850853, 31.7406540, -80.0905075, 78.3323517
6: -44.2099571, 43.5835648, -37.1664276, 36.6113968, -80.8213425, 80.7499924
7: -48.8613319, 44.6412010, -40.9768677, 37.5164680, -86.3777847, 85.6180725
8: -65.2119675, 35.4641876, -54.7549667, 30.1726189, -95.3845825, 90.2191391
9: -43.7870140, 44.0170708, -36.8862305, 37.1243057, -80.9113083, 80.9032974

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8444258, upper bound: 132.8443411
time: 8.88 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428922, upper bound: 132.8437648
time: 9.42 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -56.6306534, 45.3001862, -56.0349350, 44.8524780, -101.4831314, 101.3351212
1: -49.1118851, 39.7502708, -48.5274811, 39.3435364, -88.4553986, 88.2777481
2: -63.2026939, 39.9594460, -62.4860535, 39.5659065, -102.7686005, 102.4454956
3: -67.0065994, 34.3818665, -66.2384033, 34.1092949, -101.1158905, 100.6202698
4: -62.9855003, 46.2317924, -62.2235374, 45.8050842, -108.7905884, 108.4553299
5: -55.7565994, 43.3541565, -55.1490250, 42.8844109, -98.6410065, 98.5031738
6: -51.3029289, 50.3774796, -50.7973785, 49.8450012, -101.1479187, 101.1748581
7: -56.5052948, 50.8626900, -55.8923416, 50.2033463, -106.7086411, 106.7550201
8: -74.0791779, 42.3498650, -73.0424881, 42.2066498, -116.2858124, 115.3923492
9: -50.8275261, 50.9725647, -50.3524284, 50.4720879, -101.2996140, 101.3249817

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8532393, upper bound: 132.8523488
time: 9.29 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8506749, upper bound: 132.8513769
time: 7.11 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -53.2084122, 42.5040512, -48.5767136, 38.8977623, -92.1061707, 91.0807343
1: -46.4112015, 37.3174019, -42.2423744, 34.1155548, -80.5267563, 79.5597763
2: -59.4815254, 37.4519653, -54.2199097, 34.2871666, -93.7686691, 91.6718597
3: -63.0375786, 32.1313705, -57.4370193, 29.5219002, -92.5594788, 89.5683746
4: -59.5015450, 43.3028336, -54.1579361, 39.6359634, -99.1374969, 97.4607391
5: -52.4651146, 40.8390694, -47.8999977, 37.3298149, -89.7949295, 88.7390671
6: -48.1455231, 47.3732338, -43.9854164, 43.2251129, -91.3706131, 91.3586502
7: -53.1199150, 48.2221146, -48.4327469, 43.8848801, -97.0047913, 96.6548538
8: -70.3837280, 38.9992828, -63.9720345, 36.1114540, -106.4951782, 102.9713135
9: -47.6277809, 47.8227844, -43.5650597, 43.7583504, -91.3861313, 91.3878479

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8527510, upper bound: 132.8511443
time: 8.14 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8502601, upper bound: 132.8502601
time: 7.18 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 16.48 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 16.48
Output dim: 8, lower bound: -132.8430142, upper bound: 132.8427922
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 16.48
Output dim: 8, lower bound: -132.8415182, upper bound: 132.8422310
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 16.48
Output dim: 8, lower bound: -132.8427425, upper bound: 132.8423469
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 16.48
Output dim: 8, lower bound: -132.8412851, upper bound: 132.8417789
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.48
Output dim: 8, lower bound: -132.8502072, upper bound: 132.8486593
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.48
Output dim: 8, lower bound: -132.8478611, upper bound: 132.8476810
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.48
Output dim: 8, lower bound: -132.8499592, upper bound: 132.8479261
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.48
Output dim: 8, lower bound: -132.8475909, upper bound: 132.8470092
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 16.48
Output dim: 8, lower bound: -132.8447966, upper bound: 132.8449252
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 16.48
Output dim: 8, lower bound: -132.8432023, upper bound: 132.8443572
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 16.48
Output dim: 8, lower bound: -132.8444258, upper bound: 132.8443411
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 16.48
Output dim: 8, lower bound: -132.8428922, upper bound: 132.8437648
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.48
Output dim: 8, lower bound: -132.8532393, upper bound: 132.8523488
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.48
Output dim: 8, lower bound: -132.8506749, upper bound: 132.8513769
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.48
Output dim: 8, lower bound: -132.8527510, upper bound: 132.8511443
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.48
Output dim: 8, lower bound: -132.8502601, upper bound: 132.8502601

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -50.4868889, 40.4293785, -52.8729019, 42.3157616, -92.8026505, 93.3022690
1: -43.7711868, 35.4537125, -45.8389282, 37.1155777, -80.8867493, 81.2926407
2: -56.3119774, 35.6484184, -58.9760284, 37.3081665, -93.6201477, 94.6244507
3: -59.6262054, 30.7286587, -62.4759064, 32.1678963, -91.7940979, 93.2045517
4: -56.1376228, 41.2017860, -58.7901497, 43.1599503, -99.2975769, 99.9919357
5: -49.7173004, 38.7101364, -52.0563507, 40.4935837, -90.2108841, 90.7664719
6: -45.6992722, 44.9038925, -47.8759918, 47.0262260, -92.7254944, 92.7798691
7: -50.3047447, 45.4073105, -52.7050552, 47.5047379, -97.8094788, 98.1123657
8: -66.1955338, 37.8045578, -69.2319794, 39.5992813, -105.7948151, 107.0365295
9: -45.3032341, 45.5003281, -47.4649010, 47.6457596, -92.9489899, 92.9652252

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8457016, upper bound: 132.8443306
time: 8.91 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8452672, upper bound: 132.8437111
time: 10.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -41.3069649, 33.0906525, -47.5920601, 38.0525856, -79.3595505, 80.6826935
1: -36.0495186, 29.0537834, -41.6575813, 33.4188614, -69.4683762, 70.7113647
2: -46.1736984, 29.1269417, -53.2795105, 33.4373360, -79.6110382, 82.4064484
3: -48.8305054, 25.0287647, -56.3849754, 28.7267685, -77.5572662, 81.4137421
4: -46.3250580, 33.5345230, -53.4933434, 38.6144753, -84.9395294, 87.0278625
5: -40.8285255, 31.8617516, -47.0203705, 36.6476784, -77.4762039, 78.8821182
6: -37.2781639, 36.7930183, -42.9615059, 42.4493484, -79.7275085, 79.7545242
7: -41.1586838, 37.7159195, -47.5396080, 43.5042191, -84.6628799, 85.2555237
8: -55.2398758, 30.1466370, -63.7347336, 34.4658737, -89.7057495, 93.8813705
9: -37.0000725, 37.3124847, -42.6508865, 42.9728966, -79.9729614, 79.9633713

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428781, upper bound: 132.8431171
time: 10.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8427556, upper bound: 132.8427737
time: 11.09 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -47.2251129, 37.7547989, -45.3552933, 36.3080406, -83.5331421, 83.1100922
1: -41.2212906, 33.1352921, -39.5155983, 31.8615513, -73.0828400, 72.6508865
2: -52.7733421, 33.2453613, -50.6549072, 31.9952850, -84.7686310, 83.9002686
3: -55.8663330, 28.5474014, -53.6278839, 27.5173492, -83.3836746, 82.1752853
4: -52.8266220, 38.4034424, -50.6677895, 36.9369354, -89.7635422, 89.0712280
5: -46.5867271, 36.3158798, -44.7516937, 34.9009590, -81.4876785, 81.0675659
6: -42.6842346, 42.0428391, -41.0147476, 40.3580475, -83.0422745, 83.0575790
7: -47.0797577, 42.9108658, -45.1954384, 41.1538696, -88.2336273, 88.1063004
8: -62.7177773, 34.5555573, -60.1115417, 33.4329872, -96.1507568, 94.6670990
9: -42.2471924, 42.4884834, -40.6355896, 40.8799820, -83.1271744, 83.1240692

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8459325, upper bound: 132.8442322
time: 8.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8454759, upper bound: 132.8434961
time: 8.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -40.1745377, 32.1238823, -42.4205475, 33.9270821, -74.1016235, 74.5444260
1: -35.2618904, 28.2232933, -37.2678299, 29.8082638, -65.0701447, 65.4911194
2: -44.9721260, 28.2437935, -47.5323067, 29.7949257, -74.7670517, 75.7761002
3: -47.5922241, 24.1703854, -50.3132362, 25.5182266, -73.1104431, 74.4836197
4: -45.3269234, 32.5176735, -47.8999290, 34.3589706, -79.6858978, 80.4175949
5: -39.7534142, 31.0509624, -41.9823189, 32.7909241, -72.5443344, 73.0332794
6: -36.2209663, 35.8141937, -38.2544899, 37.8591156, -74.0800781, 74.0686722
7: -40.0596504, 36.9845581, -42.3703384, 39.0850906, -79.1447296, 79.3548965
8: -54.2561798, 28.6828651, -57.3618279, 30.2601051, -84.5162659, 86.0446777
9: -35.8664703, 36.1921768, -37.9324341, 38.2934875, -74.1599579, 74.1246109

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8430967, upper bound: 132.8429679
time: 8.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8429713, upper bound: 132.8425492
time: 9.02 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -54.4087639, 43.5189819, -52.2192459, 41.7967911, -96.2055511, 95.7382278
1: -47.2245293, 38.1920547, -45.2838058, 36.6614304, -83.8859558, 83.4758530
2: -60.7458649, 38.3780823, -58.2581635, 36.8438721, -97.5897141, 96.6362381
3: -64.3774261, 33.0212250, -61.7237778, 31.7687874, -96.1462021, 94.7450027
4: -60.5849457, 44.3736992, -58.0947762, 42.6190643, -103.2040100, 102.4684677
5: -53.5884514, 41.6848793, -51.4263916, 40.0123253, -93.6007767, 93.1112671
6: -49.2571068, 48.4040833, -47.2799797, 46.4495659, -95.7066727, 95.6840363
7: -54.2772141, 48.9797440, -52.0590172, 46.9619560, -101.2391663, 101.0387497
8: -71.4075851, 40.5171928, -68.4421005, 39.0661888, -110.4737701, 108.9592896
9: -48.8075104, 48.9911613, -46.8826370, 47.0649033, -95.8724136, 95.8737946

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8476565, upper bound: 132.8467108
time: 9.33 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8472210, upper bound: 132.8460914
time: 10.08 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -45.4618378, 36.3733330, -47.4894104, 37.9771347, -83.4389725, 83.8627319
1: -39.7175560, 31.9630470, -41.5659027, 33.3542061, -73.0717621, 73.5289459
2: -50.8770943, 32.0248375, -53.1656151, 33.3635483, -84.2406464, 85.1904526
3: -53.8658333, 27.4803009, -56.2889519, 28.6745872, -82.5404205, 83.7692490
4: -51.0366058, 36.9156494, -53.3830833, 38.5426521, -89.5792542, 90.2987061
5: -44.9297714, 35.0156250, -46.9290199, 36.5712776, -81.5010300, 81.9446411
6: -41.0474701, 40.5062408, -42.8726273, 42.3626747, -83.4101410, 83.3788681
7: -45.3749619, 41.5022736, -47.4406891, 43.4148941, -88.7898560, 88.9429626
8: -60.7417908, 33.0648232, -63.5857201, 34.4026413, -95.1444244, 96.6505356
9: -40.7254829, 41.0244484, -42.5712662, 42.8858528, -83.6113358, 83.5957184

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8447743, upper bound: 132.8455616
time: 8.48 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8446466, upper bound: 132.8451697
time: 9.12 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -50.9398766, 40.6873512, -44.8398285, 35.8986702, -86.8385391, 85.5271759
1: -44.4965820, 35.7386398, -39.0741920, 31.5052490, -76.0018311, 74.8128357
2: -56.9801178, 35.8364525, -50.0834084, 31.6262932, -88.6064148, 85.9198456
3: -60.3719978, 30.7261066, -53.0386810, 27.2023983, -87.5743866, 83.7647781
4: -57.0595016, 41.4094696, -50.1133575, 36.5084534, -93.5679550, 91.5228119
5: -50.2545776, 39.1384506, -44.2563248, 34.5161591, -84.7707367, 83.3947754
6: -46.0534134, 45.3623123, -40.5460739, 39.8989716, -85.9523849, 85.9083786
7: -50.8525658, 46.3105469, -44.6803436, 40.7206078, -91.5731735, 90.9908752
8: -67.6694794, 37.1182251, -59.4746475, 33.0146065, -100.6840820, 96.5928574
9: -45.5773277, 45.8021965, -40.1801643, 40.4189186, -85.9962311, 85.9823608

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8475901, upper bound: 132.8461927
time: 9.30 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8471678, upper bound: 132.8455177
time: 8.32 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -43.9947548, 35.1484451, -42.3024864, 33.8390312, -77.8337708, 77.4509125
1: -38.6476288, 30.9032364, -37.1620865, 29.7328415, -68.3804703, 68.0653229
2: -49.3066750, 30.9092426, -47.4008713, 29.7092686, -79.0159454, 78.3100967
3: -52.2326813, 26.4179478, -50.1982574, 25.4561119, -77.6887970, 76.6161957
4: -49.6749535, 35.6246758, -47.7741356, 34.2719269, -83.9468842, 83.3988113
5: -43.5373192, 33.9639587, -41.8754120, 32.7049942, -76.2423096, 75.8393707
6: -39.6833382, 39.2299309, -38.1522980, 37.7570686, -77.4404068, 77.3822327
7: -43.9420204, 40.4899673, -42.2570229, 38.9846077, -82.9266281, 82.7469788
8: -59.3493156, 31.3448086, -57.1950073, 30.1822014, -89.5315018, 88.5398102
9: -39.2977371, 39.6096191, -37.8401184, 38.1909828, -77.4887161, 77.4497375

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8447127, upper bound: 132.8450033
time: 8.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8446117, upper bound: 132.8446117
time: 9.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 19.73 seconds
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 19.73
Output dim: 8, lower bound: -132.8457016, upper bound: 132.8443306
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 19.73
Output dim: 8, lower bound: -132.8452672, upper bound: 132.8437111
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 19.73
Output dim: 8, lower bound: -132.8428781, upper bound: 132.8431171
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 19.73
Output dim: 8, lower bound: -132.8427556, upper bound: 132.8427737
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 19.73
Output dim: 8, lower bound: -132.8459325, upper bound: 132.8442322
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 19.73
Output dim: 8, lower bound: -132.8454759, upper bound: 132.8434961
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 19.73
Output dim: 8, lower bound: -132.8430967, upper bound: 132.8429679
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 19.73
Output dim: 8, lower bound: -132.8429713, upper bound: 132.8425492
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.73
Output dim: 8, lower bound: -132.8476565, upper bound: 132.8467108
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 19.73
Output dim: 8, lower bound: -132.8472210, upper bound: 132.8460914
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 19.73
Output dim: 8, lower bound: -132.8447743, upper bound: 132.8455616
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 19.73
Output dim: 8, lower bound: -132.8446466, upper bound: 132.8451697
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.73
Output dim: 8, lower bound: -132.8475901, upper bound: 132.8461927
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 19.73
Output dim: 8, lower bound: -132.8471678, upper bound: 132.8455177
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 19.73
Output dim: 8, lower bound: -132.8447127, upper bound: 132.8450033
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 19.73
Output dim: 8, lower bound: -132.8446117, upper bound: 132.8446117

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -51.9682083, 41.5875053, -50.4572678, 40.4024239, -92.3706207, 92.0447693
1: -45.0998001, 36.4802246, -43.7518158, 35.4263306, -80.5261230, 80.2320404
2: -58.0297508, 36.6619263, -56.2992477, 35.6039200, -93.6336670, 92.9611588
3: -61.4911079, 31.5654202, -59.6442261, 30.7146358, -92.2057419, 91.2096481
4: -57.8743858, 42.3635635, -56.1373215, 41.1690636, -99.0434418, 98.5008774
5: -51.2154922, 39.8530884, -49.7164268, 38.6895790, -89.9050598, 89.5695190
6: -47.0208092, 46.2234459, -45.6657028, 44.8748322, -91.8956375, 91.8891449
7: -51.8430328, 46.8527451, -50.3018875, 45.4298477, -97.2728653, 97.1546326
8: -68.2628555, 38.6292000, -66.1754913, 37.6981583, -105.9610138, 104.8046875
9: -46.6257591, 46.7926483, -45.3073997, 45.4771538, -92.1028900, 92.1000519

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8199709, upper bound: 132.8181555
time: 10.26 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8193684, upper bound: 132.8180331
time: 10.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -48.6366234, 38.8623848, -43.1704941, 34.5805664, -83.2171936, 82.0328827
1: -42.4894142, 34.1216049, -37.6182556, 30.3354664, -72.8248749, 71.7398605
2: -54.4142838, 34.2127419, -48.2233963, 30.4527912, -84.8670731, 82.4361343
3: -57.6546173, 29.3447227, -51.0730858, 26.2046337, -83.8592529, 80.4178009
4: -54.4975853, 39.5136185, -48.2548409, 35.1322250, -89.6298065, 87.7684326
5: -48.0179939, 37.4024582, -42.6352539, 33.2585678, -81.2765274, 80.0377045
6: -43.9419785, 43.3009491, -39.0213127, 38.4049606, -82.3469391, 82.3222427
7: -48.5534973, 44.3004761, -43.0148468, 39.2635155, -87.8170013, 87.3153076
8: -64.6918640, 35.3352737, -57.3145599, 31.7203732, -96.4122391, 92.6498260
9: -43.5188370, 43.7220383, -38.6901207, 38.9110336, -82.4298477, 82.4121552

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8277182, upper bound: 132.8257122
time: 11.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8258500, upper bound: 132.8246769
time: 11.90 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 24.69 seconds
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 24.69
Output dim: 8, lower bound: -132.8199709, upper bound: 132.8181555
IS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 24.69
Output dim: 8, lower bound: -132.8193684, upper bound: 132.8180331
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 24.69
Output dim: 8, lower bound: -132.8277182, upper bound: 132.8257122
IS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 24.69
Output dim: 8, lower bound: -132.8258500, upper bound: 132.8246769
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=145.0349884033203
rel_dist={8: [-132.9262935075506, 132.92629350749155]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9195483, upper bound: 132.9190106
time: 9.07 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9202297, upper bound: 132.9202297
time: 11.70 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 20.87 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 20.87
Output dim: 8, lower bound: -132.9195483, upper bound: 132.9190106
IS_A2, status: Status.UNKNOWN, split count: 1, time: 20.87
Output dim: 8, lower bound: -132.9202297, upper bound: 132.9202297

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -65.2420654, 52.2499275, -67.9714279, 54.4243507, -119.6664124, 120.2213440
1: -56.2837639, 45.9009018, -58.5822067, 47.8571739, -104.1409378, 104.4830933
2: -72.7158813, 46.1605721, -75.7173615, 48.1219902, -120.8378754, 121.8779221
3: -77.1514893, 39.8565636, -80.3897934, 41.5695877, -118.7210693, 120.2463531
4: -72.1226959, 53.4341507, -75.0750275, 55.7086563, -127.8313293, 128.5091858
5: -64.0869293, 49.7719688, -66.7289658, 51.8108864, -115.8978119, 116.5009308
6: -59.2973900, 58.0655937, -61.7949905, 60.5219421, -119.8193207, 119.8605804
7: -65.1676102, 57.9393730, -67.9235840, 60.2362289, -125.4038391, 125.8629608
8: -84.0978088, 49.9053078, -87.3509369, 52.1842957, -136.2821045, 137.2562408
9: -58.7354584, 58.7163086, -61.2543755, 61.2221756, -119.9576111, 119.9706802

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9013070, upper bound: 132.9033833
time: 9.04 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9106822, upper bound: 132.9102024
time: 7.89 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -69.5974808, 55.6752396, -67.3184814, 53.9052429, -123.5027161, 122.9937210
1: -60.0971375, 48.9616241, -58.0260582, 47.3964615, -107.4935989, 106.9876862
2: -77.6256561, 49.2082863, -74.9952011, 47.6526718, -125.2783279, 124.2034912
3: -82.4195938, 42.4158249, -79.6338577, 41.1705742, -123.5901642, 122.0496826
4: -77.0379486, 56.9671783, -74.3747406, 55.1715088, -132.2094574, 131.3419189
5: -68.3729553, 53.0688896, -66.0990677, 51.3261566, -119.6991043, 119.1679535
6: -63.2544403, 61.9612122, -61.1994858, 59.9397049, -123.1941452, 123.1606903
7: -69.5783997, 61.8703766, -67.2707901, 59.6875153, -129.2659149, 129.1411743
8: -89.8109360, 52.9954147, -86.5575790, 51.6510162, -141.4619446, 139.5529785
9: -62.6431046, 62.6076050, -60.6648445, 60.6324959, -123.2755966, 123.2724304

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9016507, upper bound: 132.9039076
time: 12.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.9114428, upper bound: 132.9114429
time: 12.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.44 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.44
Output dim: 8, lower bound: -132.9013070, upper bound: 132.9033833
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.44
Output dim: 8, lower bound: -132.9106822, upper bound: 132.9102024
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.44
Output dim: 8, lower bound: -132.9016507, upper bound: 132.9039076
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.44
Output dim: 8, lower bound: -132.9114428, upper bound: 132.9114429

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -57.1008301, 45.7045670, -54.0016365, 43.1950531, -100.2958832, 99.7062073
1: -49.3986740, 40.0901260, -46.7720261, 37.8820419, -87.2807083, 86.8621521
2: -63.6603165, 40.3592987, -60.1792374, 38.1618538, -101.8221741, 100.5385361
3: -67.4106140, 34.7728195, -63.6867485, 32.8273048, -100.2379150, 98.4595642
4: -63.3335953, 46.6858292, -60.0064735, 44.1235657, -107.4571609, 106.6923065
5: -56.1772690, 43.6606445, -53.1578522, 41.3239441, -97.5012131, 96.8184891
6: -51.7863312, 50.7793465, -48.9110870, 48.0152626, -99.8015747, 99.6904297
7: -56.9673309, 51.0325127, -53.8555717, 48.3899612, -105.3572922, 104.8880844
8: -74.1957245, 43.2195816, -70.3671722, 40.7053680, -114.9010925, 113.5867538
9: -51.3342171, 51.4243698, -48.5528526, 48.6978760, -100.0320892, 99.9772186

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8714168, upper bound: 132.8744113
time: 8.87 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8707546, upper bound: 132.8733950
time: 10.07 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -61.0224609, 48.8749084, -62.8007431, 50.2745018, -111.2969589, 111.6756516
1: -52.7167435, 42.8980255, -54.2704926, 44.1510696, -96.8678131, 97.1685181
2: -68.0301208, 43.1628304, -70.0071564, 44.4153175, -112.4454193, 113.1699829
3: -72.1226349, 37.2442245, -74.2465897, 38.3098831, -110.4325027, 111.4908142
4: -67.5546875, 49.9626274, -69.5640259, 51.4334641, -118.9881516, 119.5266571
5: -59.9957542, 46.6107445, -61.7426376, 47.9544640, -107.9502029, 108.3533783
6: -55.4203224, 54.2934875, -57.0444221, 55.8920021, -111.3123169, 111.3379059
7: -60.9248772, 54.3599777, -62.7276993, 55.9449234, -116.8697968, 117.0876770
8: -78.9632645, 46.4673653, -81.2265625, 47.7950363, -126.7583008, 127.6939163
9: -54.9071922, 54.9369354, -56.5407562, 56.5688744, -111.4760666, 111.4776917

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8849061, upper bound: 132.8860639
time: 9.01 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8840611, upper bound: 132.8838975
time: 8.27 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -61.4557495, 49.1387062, -53.4329300, 42.7434998, -104.1992340, 102.5716324
1: -53.2178192, 43.1509094, -46.2874527, 37.4818268, -90.6996460, 89.4383469
2: -68.5772552, 43.4090805, -59.5521317, 37.7547531, -106.3319931, 102.9612122
3: -72.6819153, 37.3341980, -63.0336533, 32.4778557, -105.1597672, 100.3678436
4: -68.2540359, 50.2247658, -59.3952026, 43.6547432, -111.9087830, 109.6199646
5: -60.4749222, 46.9577332, -52.6110611, 40.8999557, -101.3748779, 99.5687943
6: -55.7508736, 54.6783752, -48.3905716, 47.5106544, -103.2615280, 103.0689468
7: -61.3844261, 54.9726601, -53.2891617, 47.9120064, -109.2964249, 108.2618256
8: -79.9205017, 46.2970505, -69.6727905, 40.2415962, -120.1620865, 115.9698334
9: -55.2411652, 55.3099136, -48.0395126, 48.1855125, -103.4266815, 103.3494263

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8717553, upper bound: 132.8750419
time: 9.18 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8711257, upper bound: 132.8739732
time: 8.55 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -65.2793045, 52.2285042, -62.1162033, 49.7284698, -115.0077744, 114.3446808
1: -56.4471169, 45.8883057, -53.6884384, 43.6663551, -100.1134720, 99.5767441
2: -72.8317108, 46.1432648, -69.2513885, 43.9246140, -116.7563248, 115.3946533
3: -77.2774353, 39.7414894, -73.4495392, 37.8893585, -115.1667862, 113.1910248
4: -72.3687592, 53.4147797, -68.8278580, 50.8697739, -123.2385330, 122.2426300
5: -64.1904373, 49.8379326, -61.0803413, 47.4461670, -111.6365967, 110.9182434
6: -59.2941284, 58.1014862, -56.4174614, 55.2822990, -114.5764236, 114.5189362
7: -65.2400436, 58.2120667, -62.0421295, 55.3698158, -120.6098480, 120.2541809
8: -84.5643005, 49.4761848, -80.3962326, 47.2349358, -131.7992249, 129.8724213
9: -58.7278023, 58.7366180, -55.9216232, 55.9513855, -114.6791840, 114.6582413

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8717553, upper bound: 132.8750419
time: 10.00 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8843539, upper bound: 132.8843539
time: 8.54 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.68 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.68
Output dim: 8, lower bound: -132.8714168, upper bound: 132.8744113
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.68
Output dim: 8, lower bound: -132.8707546, upper bound: 132.8733950
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.68
Output dim: 8, lower bound: -132.8849061, upper bound: 132.8860639
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.68
Output dim: 8, lower bound: -132.8840611, upper bound: 132.8838975
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.68
Output dim: 8, lower bound: -132.8717553, upper bound: 132.8750419
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.68
Output dim: 8, lower bound: -132.8711257, upper bound: 132.8739732
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.68
Output dim: 8, lower bound: -132.8717553, upper bound: 132.8750419
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.68
Output dim: 8, lower bound: -132.8843539, upper bound: 132.8843539

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -50.2539597, 40.2267914, -51.2617683, 41.0090752, -91.2630310, 91.4885559
1: -43.5888596, 35.2617760, -44.4476433, 35.9459915, -79.5348358, 79.7094116
2: -56.0393066, 35.4951591, -57.1232681, 36.2182579, -92.2575684, 92.6184158
3: -59.3055725, 30.5258522, -60.4503517, 31.1184444, -90.4240036, 90.9762039
4: -55.9089355, 40.9967728, -57.0261726, 41.8460503, -97.7549667, 98.0229492
5: -49.5025291, 38.5403061, -50.4916687, 39.2741241, -88.7766571, 89.0319748
6: -45.4752083, 44.6843948, -46.3863258, 45.5715256, -91.0467224, 91.0707245
7: -50.0718002, 45.2194519, -51.0951805, 46.0604630, -96.1322556, 96.3146362
8: -65.8892670, 37.5947685, -67.0401459, 38.4533005, -104.3425674, 104.6349030
9: -45.0962563, 45.2895851, -46.0548096, 46.2393265, -91.3355637, 91.3443909

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8712953, upper bound: 132.8744113
time: 9.14 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8712953, upper bound: 132.8744113
time: 10.44 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -47.1375694, 37.6711273, -44.9313354, 35.9556770, -83.0932388, 82.6024628
1: -41.1560020, 33.0409851, -39.1115456, 31.5033951, -72.6593933, 72.1525192
2: -52.6570969, 33.1936951, -50.1000061, 31.7365761, -84.3936768, 83.2936859
3: -55.7107430, 28.4215984, -52.9794769, 27.1829243, -82.8936691, 81.4010620
4: -52.7425194, 38.3122253, -50.1564369, 36.5897751, -89.3322906, 88.4686584
5: -46.5149689, 36.2477303, -44.3367538, 34.5487366, -81.0637054, 80.5844650
6: -42.5866470, 41.9489479, -40.5994377, 39.9472198, -82.5338593, 82.5483856
7: -46.9862480, 42.8428535, -44.7478676, 40.6896858, -87.6759338, 87.5907135
8: -62.5765343, 34.4603615, -59.3349724, 33.2529144, -95.8294449, 93.7953339
9: -42.1685982, 42.3981476, -40.2911186, 40.5207481, -82.6893387, 82.6892548

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8706315, upper bound: 132.8733950
time: 8.05 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8706315, upper bound: 132.8733950
time: 10.75 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -53.7327271, 43.0324173, -59.7264595, 47.8057709, -101.5384979, 102.7588806
1: -46.5251122, 37.7316437, -51.6582565, 41.9660187, -88.4911346, 89.3899002
2: -59.9055557, 37.9493790, -66.5770493, 42.2121620, -102.1177216, 104.5264206
3: -63.4733810, 32.7288475, -70.5925903, 36.4036369, -99.8770065, 103.3214340
4: -59.6426544, 43.9025841, -66.2197342, 48.8759766, -108.5186310, 110.1223145
5: -52.8796120, 41.1430817, -58.7394905, 45.6461105, -98.5257263, 99.8825684
6: -48.6853371, 47.7854614, -54.1992798, 53.1414299, -101.8267593, 101.9847412
7: -53.5550308, 48.1579857, -59.6166534, 53.3257446, -106.8807678, 107.7746429
8: -70.1215820, 40.4637451, -77.4989777, 45.2562218, -115.3777924, 117.9627228
9: -48.2485886, 48.3928833, -53.7266235, 53.8044853, -102.0530701, 102.1195068

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8846711, upper bound: 132.8860639
time: 8.59 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8846711, upper bound: 132.8860639
time: 9.10 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -50.4350739, 40.3323402, -52.8024559, 42.2707787, -92.7058487, 93.1347961
1: -43.9358597, 35.3805656, -45.8052750, 37.0647087, -81.0005493, 81.1858368
2: -56.3250198, 35.5297699, -58.8791962, 37.2787704, -93.6037903, 94.4089661
3: -59.6502953, 30.5464439, -62.3996811, 32.1402855, -91.7905807, 92.9461212
4: -56.2926331, 41.0762711, -58.7188416, 43.1350555, -99.4276657, 99.7951126
5: -49.7125397, 38.7295494, -51.9964676, 40.4688110, -90.1813431, 90.7260132
6: -45.6431198, 44.8951988, -47.8443146, 46.9723930, -92.6155090, 92.7394943
7: -50.2952080, 45.6288528, -52.6585426, 47.4416466, -97.7368546, 98.2873840
8: -66.5998459, 37.1976357, -69.0860672, 39.5786209, -106.1784592, 106.2836838
9: -45.1475220, 45.3560181, -47.3979988, 47.5565643, -92.7040863, 92.7540131

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8838576, upper bound: 132.8838974
time: 7.88 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8838576, upper bound: 132.8838975
time: 7.16 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -54.4164429, 43.5075722, -50.7195282, 40.5776024, -94.9940414, 94.2270889
1: -47.2455826, 38.1716957, -43.9867249, 35.5674438, -82.8130264, 82.1584167
2: -60.7395325, 38.3923302, -56.5277214, 35.8301506, -96.5696793, 94.9200516
3: -64.3466034, 32.9685822, -59.8285027, 30.7858315, -95.1324310, 92.7970886
4: -60.6191177, 44.3695221, -56.4446068, 41.3979263, -102.0170441, 100.8141327
5: -53.6078110, 41.6931114, -49.9718704, 38.8691711, -92.4769745, 91.6649780
6: -49.2525406, 48.4003525, -45.8921852, 45.0924873, -94.3450241, 94.2925415
7: -54.2865791, 48.9952393, -50.5561867, 45.6057663, -99.8923416, 99.5514221
8: -71.3899155, 40.4996071, -66.3776550, 38.0102768, -109.4001923, 106.8772583
9: -48.8178978, 48.9929008, -45.5661659, 45.7521286, -94.5700226, 94.5590591

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8467396, upper bound: 132.8470274
time: 10.11 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8436772, upper bound: 132.8458574
time: 10.05 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -51.0382080, 40.7462959, -44.4511833, 35.5744324, -86.6126328, 85.1974792
1: -44.5859184, 35.7745781, -38.7034607, 31.1722775, -75.7581940, 74.4780273
2: -57.0667877, 35.9114571, -49.5721283, 31.3932266, -88.4600143, 85.4835739
3: -60.4373436, 30.7194653, -52.4310684, 26.8879528, -87.3252869, 83.1505280
4: -57.1789169, 41.4739532, -49.6407394, 36.1927376, -93.3716354, 91.1146927
5: -50.3621025, 39.2061386, -43.8754997, 34.1906395, -84.5527267, 83.0816345
6: -46.1214867, 45.4341507, -40.1614227, 39.5222549, -85.6437378, 85.5955734
7: -50.9450760, 46.3954163, -44.2672272, 40.2878876, -91.2329559, 90.6626434
8: -67.7466507, 37.1712379, -58.7443390, 32.8622398, -100.6088867, 95.9155731
9: -45.6640892, 45.8768921, -39.8630714, 40.0873260, -85.7514038, 85.7399597

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8459944, upper bound: 132.8458426
time: 9.40 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8430730, upper bound: 132.8446861
time: 7.68 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -57.8577728, 46.2811813, -59.0421486, 47.2597923, -105.1175613, 105.3233337
1: -50.1480408, 40.6209641, -51.0757790, 41.4813080, -91.6293335, 91.6967316
2: -64.5633698, 40.8296814, -65.8215942, 41.7215385, -106.2849121, 106.6512680
3: -68.4665985, 35.1450005, -69.7956467, 35.9823685, -104.4489670, 104.9406433
4: -64.3077469, 47.2461548, -65.4845886, 48.3104820, -112.6182251, 112.7307434
5: -56.9506645, 44.2729721, -58.0772743, 45.1374664, -102.0881271, 102.3502502
6: -52.4314613, 51.4708939, -53.5718651, 52.5317650, -104.9632263, 105.0427551
7: -57.7379837, 51.9012375, -58.9310112, 52.7510986, -110.4890823, 110.8322372
8: -75.5740509, 43.3484039, -76.6698227, 44.6973610, -120.2714081, 120.0182266
9: -51.9399109, 52.0696716, -53.1072044, 53.1876755, -105.1275864, 105.1768570

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8559836, upper bound: 132.8541569
time: 10.65 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8510708, upper bound: 132.8523759
time: 9.09 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -54.2846298, 43.3655930, -52.2084122, 41.7997093, -96.0843353, 95.5739822
1: -47.3150520, 38.0724564, -45.3017273, 36.6532211, -83.9682770, 83.3741837
2: -60.6693535, 38.2110748, -58.2268677, 36.8556137, -97.5249634, 96.4379425
3: -64.3139648, 32.8062782, -61.7146416, 31.7765427, -96.0905075, 94.5209198
4: -60.6599121, 44.1908417, -58.0821152, 42.6459312, -103.3058472, 102.2729416
5: -53.5099716, 41.6426888, -51.4251785, 40.0306320, -93.5405960, 93.0678711
6: -49.1339417, 48.3282890, -47.3030052, 46.4482994, -95.5822296, 95.6312943
7: -54.1942368, 49.1305542, -52.0675354, 46.9468765, -101.1411133, 101.1980896
8: -71.6937714, 39.8725357, -68.3632889, 39.0946655, -110.7884293, 108.2358093
9: -48.5960197, 48.7836418, -46.8658371, 47.0255051, -95.6215134, 95.6494751

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8550782, upper bound: 132.8520232
time: 9.11 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8502790, upper bound: 132.8502790
time: 8.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.90 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 8, lower bound: -132.8712953, upper bound: 132.8744113
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 8, lower bound: -132.8712953, upper bound: 132.8744113
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 8, lower bound: -132.8706315, upper bound: 132.8733950
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 8, lower bound: -132.8706315, upper bound: 132.8733950
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 8, lower bound: -132.8846711, upper bound: 132.8860639
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 8, lower bound: -132.8846711, upper bound: 132.8860639
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 8, lower bound: -132.8838576, upper bound: 132.8838974
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 8, lower bound: -132.8838576, upper bound: 132.8838975
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 18.90
Output dim: 8, lower bound: -132.8467396, upper bound: 132.8470274
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 18.90
Output dim: 8, lower bound: -132.8436772, upper bound: 132.8458574
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 18.90
Output dim: 8, lower bound: -132.8459944, upper bound: 132.8458426
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 18.90
Output dim: 8, lower bound: -132.8430730, upper bound: 132.8446861
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 8, lower bound: -132.8559836, upper bound: 132.8541569
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 8, lower bound: -132.8510708, upper bound: 132.8523759
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 8, lower bound: -132.8550782, upper bound: 132.8520232
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.90
Output dim: 8, lower bound: -132.8502790, upper bound: 132.8502790

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -50.2539597, 40.2267914, -49.0171776, 39.2148781, -89.4688416, 89.2439728
1: -43.5888596, 35.2617760, -42.5621643, 34.3556938, -77.9445496, 77.8239441
2: -56.0393066, 35.4951591, -54.6716881, 34.6273880, -90.6666946, 90.1668472
3: -59.3055725, 30.5258522, -57.7926865, 29.6926746, -88.9982452, 88.3185272
4: -55.9089355, 40.9967728, -54.5957031, 39.9676437, -95.8765717, 95.5924759
5: -49.5025291, 38.5403061, -48.3170013, 37.6017761, -87.1043091, 86.8573074
6: -45.4752083, 44.6843948, -44.3332367, 43.5742378, -89.0494461, 89.0176239
7: -50.0718002, 45.2194519, -48.8420181, 44.1820107, -94.2538147, 94.0614700
8: -65.8892670, 37.5947685, -64.3683243, 36.5773087, -102.4665756, 101.9630814
9: -45.0962563, 45.2895851, -43.9887428, 44.1915894, -89.2878418, 89.2783279

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8437650, upper bound: 132.8480264
time: 8.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8416919, upper bound: 132.8435569
time: 8.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -50.2539597, 40.2267914, -53.0647049, 42.4055328, -92.6594849, 93.2914963
1: -43.5888596, 35.2617760, -46.1175461, 37.1913452, -80.7801895, 81.3793182
2: -56.0393066, 35.4951591, -59.2440720, 37.4482880, -93.4875946, 94.7392273
3: -59.3055725, 30.5258522, -62.6909485, 32.0827827, -91.3883514, 93.2167892
4: -55.9089355, 40.9967728, -59.1844826, 43.2491684, -99.1581039, 100.1812592
5: -49.5025291, 38.5403061, -52.3069000, 40.6696930, -90.1722260, 90.8472061
6: -45.4752083, 44.6843948, -48.0036659, 47.1920471, -92.6672516, 92.6880493
7: -50.0718002, 45.2194519, -52.9455605, 47.8523865, -97.9241638, 98.1650085
8: -65.8892670, 37.5947685, -69.7130127, 39.4148369, -105.3041077, 107.3077774
9: -45.0962563, 45.2895851, -47.6113739, 47.8004684, -92.8967133, 92.9009552

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8437650, upper bound: 132.8481498
time: 8.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8416919, upper bound: 132.8435589
time: 8.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -47.1375694, 37.6711273, -42.8798943, 34.3179474, -81.4555206, 80.5510254
1: -41.1560020, 33.0409851, -37.3930626, 30.0649109, -71.2209167, 70.4340515
2: -52.6570969, 33.1936951, -47.8591003, 30.2862873, -82.9433823, 81.0527954
3: -55.7107430, 28.4215984, -50.5597878, 25.8792248, -81.5899658, 78.9813690
4: -52.7425194, 38.3122253, -47.9411087, 34.8701973, -87.6127167, 86.2533264
5: -46.5149689, 36.2477303, -42.3450279, 33.0190239, -79.5339966, 78.5927582
6: -42.5866470, 41.9489479, -38.7252312, 38.1264763, -80.7131195, 80.6741638
7: -46.9862480, 42.8428535, -42.6868935, 38.9751740, -85.9614258, 85.5297470
8: -62.5765343, 34.4603615, -56.8841515, 31.5390968, -94.1156311, 91.3445129
9: -42.1685982, 42.3981476, -38.4098663, 38.6457596, -80.8143616, 80.8080063

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428097, upper bound: 132.8467530
time: 8.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8411554, upper bound: 132.8425839
time: 10.37 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -47.1375694, 37.6711273, -46.7936935, 37.3998108, -84.5373764, 84.4648209
1: -41.1560020, 33.0409851, -40.8303146, 32.8001785, -73.9561768, 73.8712921
2: -52.6570969, 33.1936951, -52.2822075, 33.0093803, -85.6664734, 85.4758987
3: -55.7107430, 28.4215984, -55.2967796, 28.1885262, -83.8992691, 83.7183685
4: -52.7425194, 38.3122253, -52.3845062, 38.0421524, -90.7846680, 90.6967239
5: -46.5149689, 36.2477303, -46.2077408, 35.9868813, -82.5018463, 82.4554443
6: -42.5866470, 41.9489479, -42.2699242, 41.6197166, -84.2063599, 84.2188644
7: -46.9862480, 42.8428535, -46.6581802, 42.5341377, -89.5203857, 89.5010223
8: -62.5765343, 34.4603615, -62.0654907, 34.2678032, -96.8443375, 96.5258484
9: -42.1685982, 42.3981476, -41.9125900, 42.1343079, -84.3028946, 84.3107376

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8428097, upper bound: 132.8469579
time: 8.83 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8411554, upper bound: 132.8426294
time: 8.10 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -53.7327271, 43.0324173, -57.1647758, 45.7594490, -99.4921722, 100.1971817
1: -46.5251122, 37.7316437, -49.4973450, 40.1317825, -86.6568680, 87.2289734
2: -59.9055557, 37.9493790, -63.7609177, 40.3745193, -100.2800751, 101.7102814
3: -63.4733810, 32.7288475, -67.5517273, 34.7938271, -98.2672043, 100.2805710
4: -59.6426544, 43.9025841, -63.4421463, 46.7302132, -106.3728638, 107.3447189
5: -52.8796120, 41.1430817, -56.2516174, 43.7319183, -96.6115265, 97.3946991
6: -48.6853371, 47.7854614, -51.8432846, 50.8421593, -99.5274963, 99.6287460
7: -53.5550308, 48.1579857, -57.0256767, 51.1683350, -104.7233658, 105.1836624
8: -70.1215820, 40.4637451, -74.4449310, 43.1149940, -113.2365723, 114.9086761
9: -48.2485886, 48.3928833, -51.3605537, 51.4608345, -99.7094269, 99.7534332

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8499257, upper bound: 132.8539750
time: 9.99 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8475561, upper bound: 132.8483348
time: 8.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -53.7327271, 43.0324173, -61.1696472, 48.9161491, -102.6488800, 104.2020569
1: -46.5251122, 37.7316437, -53.0196838, 42.9443893, -89.4694977, 90.7513199
2: -59.9055557, 37.9493790, -68.2872391, 43.1788330, -103.0843887, 106.2366180
3: -63.4733810, 32.7288475, -72.4032211, 37.1398849, -100.6132584, 105.1320572
4: -59.6426544, 43.9025841, -67.9854050, 49.9829369, -109.6255798, 111.8879852
5: -52.8796120, 41.1430817, -60.2059898, 46.7706528, -99.6502609, 101.3490753
6: -48.6853371, 47.7854614, -55.4862595, 54.4305267, -103.1158600, 103.2717209
7: -53.5550308, 48.1579857, -61.0934830, 54.8066330, -108.3616638, 109.2514648
8: -70.1215820, 40.4637451, -79.7391586, 45.9245415, -116.0461273, 120.2029037
9: -48.2485886, 48.3928833, -54.9538956, 55.0362587, -103.2848434, 103.3467789

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8499257, upper bound: 132.8543935
time: 10.29 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8475561, upper bound: 132.8484193
time: 6.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -50.4350739, 40.3323402, -50.5249481, 40.4512482, -90.8863220, 90.8572845
1: -43.9358597, 35.3805656, -43.8944016, 35.4648056, -79.4006653, 79.2749634
2: -56.3250198, 35.5297699, -56.3984909, 35.6671410, -91.9921570, 91.9282608
3: -59.6502953, 30.5464439, -59.7075005, 30.7037926, -90.3540878, 90.2539368
4: -56.2926331, 41.0762711, -56.2638626, 41.2341690, -97.5268021, 97.3401337
5: -49.7125397, 38.7295494, -49.7889481, 38.7818985, -88.4944305, 88.5184937
6: -45.6431198, 44.8951988, -45.7646942, 44.9504509, -90.5935669, 90.6598816
7: -50.2952080, 45.6288528, -50.3803711, 45.5427742, -95.8379745, 96.0092239
8: -66.5998459, 37.1976357, -66.3833008, 37.6770287, -104.2768555, 103.5809326
9: -45.1475220, 45.3560181, -45.3076782, 45.4889183, -90.6364365, 90.6636963

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8489598, upper bound: 132.8522767
time: 10.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8469287, upper bound: 132.8469287
time: 7.03 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -50.4350739, 40.3323402, -54.4046249, 43.5089951, -93.9440613, 94.7369690
1: -43.9358597, 35.3805656, -47.3052788, 38.1765862, -82.1124344, 82.6858444
2: -56.3250198, 35.5297699, -60.7776222, 38.3726425, -94.6976624, 96.3073883
3: -59.6502953, 30.5464439, -64.4063492, 32.9807167, -92.6310120, 94.9527893
4: -56.2926331, 41.0762711, -60.6640282, 44.3746796, -100.6672974, 101.7402954
5: -49.7125397, 38.7295494, -53.6150436, 41.7204666, -91.4330063, 92.3445892
6: -45.6431198, 44.8951988, -49.2847900, 48.4147453, -94.0578613, 94.1799774
7: -50.2952080, 45.6288528, -54.3109665, 49.0691071, -99.3643188, 99.9398193
8: -66.5998459, 37.1976357, -71.5152283, 40.3838959, -106.9837418, 108.7128601
9: -45.1475220, 45.3560181, -48.7837067, 48.9457169, -94.0932388, 94.1397247

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8489598, upper bound: 132.8527372
time: 8.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8469287, upper bound: 132.8471781
time: 6.03 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -56.9317894, 45.5397530, -55.0292282, 44.0356293, -100.9674225, 100.5689774
1: -49.3612442, 39.9689178, -47.6596870, 38.6443443, -88.0055847, 87.6286011
2: -63.5385818, 40.1668701, -61.3708038, 38.8367424, -102.3753128, 101.5376740
3: -67.3703079, 34.5789299, -65.0429153, 33.5214653, -100.8917694, 99.6218414
4: -63.3067856, 46.4732971, -61.1286812, 44.9576378, -108.2644196, 107.6019745
5: -56.0458069, 43.5752258, -54.1553917, 42.1045609, -98.1503677, 97.7306061
6: -51.5776634, 50.6462708, -49.8636894, 48.9524956, -100.5301590, 100.5099640
7: -56.8051147, 51.1158676, -54.8810577, 49.3319969, -106.1371155, 105.9969254
8: -74.4592896, 42.5846977, -71.8327789, 41.3857956, -115.8450775, 114.4174652
9: -51.0963402, 51.2435226, -49.4447556, 49.5980453, -100.6943817, 100.6882629

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8507371, upper bound: 132.8488887
time: 10.07 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8499458, upper bound: 132.8477083
time: 9.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -49.2110329, 39.3648148, -49.8919411, 39.8983231, -89.1093597, 89.2567520
1: -42.8720932, 34.5824814, -43.6002617, 35.0393906, -77.9114838, 78.1827393
2: -55.0168152, 34.6821709, -55.8213501, 35.0648193, -90.0816345, 90.5035248
3: -58.2745323, 29.8208561, -59.1173477, 30.1868477, -88.4613800, 88.9382019
4: -55.0465546, 40.0307846, -55.9684372, 40.5443115, -95.5908661, 95.9992218
5: -48.5560913, 37.8186531, -49.2602539, 38.3631821, -86.9192657, 87.0789032
6: -44.4873123, 43.8196945, -45.0806961, 44.4892731, -88.9765778, 88.9003830
7: -49.1176643, 44.6459503, -49.8529205, 45.4417343, -94.5593796, 94.4988632
8: -65.2497406, 36.1756859, -66.4867935, 36.4051933, -101.6549301, 102.6624756
9: -44.1148987, 44.3754768, -44.7616196, 45.0543137, -89.1691971, 89.1370926

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8452054, upper bound: 132.8467363
time: 9.19 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8449503, upper bound: 132.8459214
time: 9.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -53.3706970, 42.6334915, -48.3691063, 38.7217674, -92.0924683, 91.0025940
1: -46.5433083, 37.4361877, -42.0521240, 33.9763794, -80.5196838, 79.4883118
2: -59.6611862, 37.5612106, -53.9851151, 34.1256294, -93.7868195, 91.5463257
3: -63.2380753, 32.2401161, -57.1900253, 29.4059792, -92.6440506, 89.4301453
4: -59.6761169, 43.4273605, -53.9337540, 39.4433441, -99.1194534, 97.3611145
5: -52.6187439, 40.9577789, -47.6857185, 37.1516304, -89.7703705, 88.6434937
6: -48.2917480, 47.5173988, -43.7732697, 43.0386124, -91.3303604, 91.2906647
7: -53.2805481, 48.3601532, -48.2190056, 43.7033234, -96.9838715, 96.5791626
8: -70.6006546, 39.1134720, -63.7553787, 35.9229851, -106.5236282, 102.8688354
9: -47.7689056, 47.9694366, -43.3865166, 43.6065483, -91.3754578, 91.3559418

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8503772, upper bound: 132.8476964
time: 9.00 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8496385, upper bound: 132.8464152
time: 11.20 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -47.1636963, 37.6770935, -45.0419693, 36.0323563, -83.1960526, 82.7190552
1: -41.3222237, 33.1137466, -39.4921227, 31.6479263, -72.9701538, 72.6058578
2: -52.8112068, 33.1505928, -50.4440880, 31.6447945, -84.4560013, 83.5946808
3: -55.9511948, 28.3969078, -53.4121552, 27.1733589, -83.1245499, 81.8090439
4: -53.0481377, 38.2556610, -50.7294807, 36.5366859, -89.5848236, 88.9851379
5: -46.6116714, 36.3219795, -44.5483856, 34.7482719, -81.3599396, 80.8703613
6: -42.5874939, 42.0331039, -40.6561584, 40.1899681, -82.7774658, 82.6892548
7: -47.0979729, 43.1537018, -44.9984474, 41.3175812, -88.4155579, 88.1521454
8: -63.1684380, 33.9644775, -60.5514603, 32.4377823, -95.6062164, 94.5159378
9: -42.1613045, 42.4404945, -40.3427048, 40.6725159, -82.8338165, 82.7832031

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8448254, upper bound: 132.8453752
time: 7.33 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8446233, upper bound: 132.8446233
time: 7.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 16.38 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8437650, upper bound: 132.8480264
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8416919, upper bound: 132.8435569
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8437650, upper bound: 132.8481498
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8416919, upper bound: 132.8435589
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8428097, upper bound: 132.8467530
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8411554, upper bound: 132.8425839
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8428097, upper bound: 132.8469579
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8411554, upper bound: 132.8426294
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8499257, upper bound: 132.8539750
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8475561, upper bound: 132.8483348
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8499257, upper bound: 132.8543935
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8475561, upper bound: 132.8484193
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8489598, upper bound: 132.8522767
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8469287, upper bound: 132.8469287
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8489598, upper bound: 132.8527372
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8469287, upper bound: 132.8471781
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8507371, upper bound: 132.8488887
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8499458, upper bound: 132.8477083
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8452054, upper bound: 132.8467363
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8449503, upper bound: 132.8459214
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8503772, upper bound: 132.8476964
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8496385, upper bound: 132.8464152
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8448254, upper bound: 132.8453752
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.38
Output dim: 8, lower bound: -132.8446233, upper bound: 132.8446233

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -46.8422127, 37.4919434, -48.1829948, 38.5458145, -85.3880310, 85.6749420
1: -40.6911316, 32.8795395, -41.8525505, 33.7724953, -74.4636230, 74.7320862
2: -52.2585106, 33.0697784, -53.7447090, 34.0330963, -86.2916107, 86.8144836
3: -55.2847977, 28.4183617, -56.8078651, 29.1773529, -84.4621506, 85.2262268
4: -52.2082405, 38.1525078, -53.6877823, 39.2698097, -91.4780502, 91.8402863
5: -46.1786423, 35.9732857, -47.5030556, 36.9724960, -83.1511307, 83.4763336
6: -42.3358383, 41.6505737, -43.5643616, 42.8325310, -85.1683655, 85.2149353
7: -46.6472778, 42.3238106, -48.0014496, 43.4711800, -90.1184540, 90.3252563
8: -61.7714729, 34.7865067, -63.3570442, 35.8907471, -97.6622162, 98.1435547
9: -42.0035210, 42.2476463, -43.2324677, 43.4448853, -85.4483871, 85.4801025

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8392390, upper bound: 132.8431836
time: 9.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8383277, upper bound: 132.8427558
time: 9.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -46.8422127, 37.4919434, -52.1844215, 41.6995850, -88.5417938, 89.6763611
1: -40.6911316, 32.8795395, -45.3699570, 36.5740547, -77.2651825, 78.2494965
2: -52.2585106, 33.0697784, -58.2689133, 36.8225708, -89.0810699, 91.3386917
3: -55.2847977, 28.4183617, -61.6508102, 31.5380802, -86.8228760, 90.0691681
4: -52.2082405, 38.1525078, -58.2304840, 42.5122223, -94.7204514, 96.3829956
5: -46.1786423, 35.9732857, -51.4483185, 40.0067139, -86.1853485, 87.4216003
6: -42.3358383, 41.6505737, -47.1937561, 46.4081268, -88.7439651, 88.8443146
7: -46.6472778, 42.3238106, -52.0611000, 47.1050682, -93.7523346, 94.3849106
8: -61.7714729, 34.7865067, -68.6521301, 38.6892853, -100.4607544, 103.4386368
9: -42.0035210, 42.2476463, -46.8129959, 47.0143814, -89.0178909, 89.0606384

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8397622, upper bound: 132.8432793
time: 10.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8388927, upper bound: 132.8428607
time: 10.56 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -50.2166367, 40.2130165, -56.2494736, 45.0243034, -95.2409363, 96.4624939
1: -43.5367165, 35.2719231, -48.7176514, 39.4854088, -83.0221252, 83.9895782
2: -56.0131760, 35.4513359, -62.7456436, 39.7163506, -95.7295227, 98.1969757
3: -59.3171692, 30.5730629, -66.4684067, 34.2338562, -93.5510254, 97.0414734
4: -55.8431549, 40.9689255, -62.4487839, 45.9663887, -101.8095398, 103.4177094
5: -49.4494858, 38.5029755, -55.3571510, 43.0398483, -92.4893112, 93.8601227
6: -45.4480057, 44.6656151, -50.9978294, 50.0271034, -95.4751129, 95.6634445
7: -50.0323753, 45.1761627, -56.1026917, 50.3876152, -100.4199829, 101.2788467
8: -65.8849792, 37.5723991, -73.3401413, 42.3613205, -108.2462921, 110.9125366
9: -45.0544968, 45.2619553, -50.5255814, 50.6434326, -95.6979065, 95.7875366

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8475561, upper bound: 132.8483410
time: 9.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -132.8475561, upper bound: 132.8483410
time: 8.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -45.3783073, 36.2966347, -48.2208214, 38.5987206, -83.9770279, 84.5174561
1: -39.7334404, 31.8802032, -41.9596825, 33.8734322, -73.6068497, 73.8398895
2: -50.8027954, 31.8851929, -53.8772049, 34.0051689, -84.8079681, 85.7623901
3: -53.7527008, 27.3946762, -56.9984131, 29.2739601, -83.0266418, 84.3930893
4: -51.0400467, 36.7872238, -53.8382301, 39.2657051, -90.3057556, 90.6254501
5: -44.8457184, 34.9818573, -47.5662575, 37.0480728, -81.8937759, 82.5481110
6: -40.9368896, 40.4831009, -43.6183205, 42.9213638, -83.8582535, 84.1014252
7: -45.3126106, 41.5480919, -48.0954437, 43.6509781, -88.9635849, 89.6435165
8: -60.9381638, 32.7650146, -63.7586021, 35.7004128, -96.6385803, 96.5236206
9: -40.6327095, 40.9848671, -43.2556725, 43.4952126, -84.1279144, 84.2405396

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 175

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8435094, upper bound: 132.8436457
time: 8.04 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -132.8426472, upper bound: 132.8433719
time: 6.92 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 16.12 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 16.12
Output dim: 8, lower bound: -132.8392390, upper bound: 132.8431836
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 16.12
Output dim: 8, lower bound: -132.8383277, upper bound: 132.8427558
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 16.12
Output dim: 8, lower bound: -132.8397622, upper bound: 132.8432793
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 16.12
Output dim: 8, lower bound: -132.8388927, upper bound: 132.8428607
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.12
Output dim: 8, lower bound: -132.8475561, upper bound: 132.8483410
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 16.12
Output dim: 8, lower bound: -132.8475561, upper bound: 132.8483410
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 16.12
Output dim: 8, lower bound: -132.8435094, upper bound: 132.8436457
IS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 16.12
Output dim: 8, lower bound: -132.8426472, upper bound: 132.8433719
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.12
Output dim: 8, lower bound: -132.8499257, upper bound: 132.8543935
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.12
Output dim: 8, lower bound: -132.8475561, upper bound: 132.8484193
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.12
Output dim: 8, lower bound: -132.8489598, upper bound: 132.8522767
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.12
Output dim: 8, lower bound: -132.8489598, upper bound: 132.8527372
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.12
Output dim: 8, lower bound: -132.8507371, upper bound: 132.8488887
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.12
Output dim: 8, lower bound: -132.8499458, upper bound: 132.8477083
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.12
Output dim: 8, lower bound: -132.8503772, upper bound: 132.8476964
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.12
Output dim: 8, lower bound: -132.8496385, upper bound: 132.8464152
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=145.0349884033203
rel_dist={8: [-132.9263529894642, 132.92635298946425]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 2361.84 seconds
