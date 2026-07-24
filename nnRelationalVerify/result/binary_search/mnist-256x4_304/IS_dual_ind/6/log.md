## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 107.2381207338
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-66.9629440, 52.5338936, -66.9629440, 52.5338936, -119.4968338, 119.4968414)
1: (-55.4344978, 47.0252304, -55.4344978, 47.0252304, -102.4597092, 102.4597092)
2: (-70.9811935, 43.8698730, -70.9811935, 43.8698730, -114.8510666, 114.8510666)
3: (-81.4659729, 39.4653587, -81.4659729, 39.4653587, -120.9313354, 120.9313354)
4: (-72.6269455, 55.4208984, -72.6269455, 55.4208984, -128.0478363, 128.0478363)
5: (-62.9200974, 48.7844696, -62.9200974, 48.7844696, -111.7045670, 111.7045670)
6: (-60.2321777, 60.8447495, -60.2321777, 60.8447495, -121.0769272, 121.0769196)
7: (-69.6045456, 50.5524025, -69.6045456, 50.5524025, -120.1569519, 120.1569519)
8: (-77.8515930, 52.6643562, -77.8515930, 52.6643562, -130.5159454, 130.5159454)
9: (-60.1245384, 60.0685272, -60.1245384, 60.0685272, -120.1930695, 120.1930618)

## BASE Result
execution time: IAR + LP analysis = 1.42 + 12.05 = 13.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -107.3456123, upper bound: 107.3456123


# Binary Search by BASE starts (time budget: 1986.53 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=120.15695190429688
rel_dist={7: [-107.34558116528925, 107.34558116528925]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=120.15695190429688
rel_dist={7: [-107.34546615888509, 107.34546613728932]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=120.15695190429688
rel_dist={7: [-107.34537841147622, 107.3453783970549]}

## Binary Search Result
Binary search time: 44.88 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1941.66 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3418513, upper bound: 107.3417940
time: 9.68 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3420674, upper bound: 107.3420674
time: 7.18 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.01 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.01
Output dim: 7, lower bound: -107.3418513, upper bound: 107.3417940
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.01
Output dim: 7, lower bound: -107.3420674, upper bound: 107.3420674

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -54.8241081, 43.0569534, -66.9629440, 52.5338936, -107.3580017, 110.0198975
1: -45.4302597, 38.5692978, -55.4344978, 47.0252304, -92.4554749, 94.0037842
2: -57.7149849, 35.3450928, -70.9811935, 43.8698730, -101.5848541, 106.3262787
3: -67.1298523, 31.9761009, -81.4659729, 39.4653587, -106.5952148, 113.4420547
4: -59.6581039, 45.5115700, -72.6269455, 55.4208984, -115.0789871, 118.1385193
5: -51.4531670, 39.7658691, -62.9200974, 48.7844696, -100.2376251, 102.6859665
6: -49.0709686, 50.1650696, -60.2321777, 60.8447495, -109.9157181, 110.3972397
7: -57.5313797, 40.3114319, -69.6045456, 50.5524025, -108.0837860, 109.9159775
8: -63.1841316, 42.7644730, -77.8515930, 52.6643562, -115.8484879, 120.6160660
9: -49.1056328, 49.1908188, -60.1245384, 60.0685272, -109.1741486, 109.3153534

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3416339, upper bound: 107.3416338
time: 11.06 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3416339, upper bound: 107.3417877
time: 9.70 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -59.6072922, 46.7624054, -65.1414566, 51.0990639, -110.7063599, 111.9038620
1: -49.3685532, 41.8875961, -53.9287491, 45.7521210, -95.1206665, 95.8163376
2: -62.8696785, 38.5685043, -68.9852448, 42.5909882, -105.4606628, 107.5537491
3: -72.8384476, 34.8074989, -79.3149414, 38.3396835, -111.1781235, 114.1224289
4: -64.8151855, 49.4113007, -70.6797714, 53.9264908, -118.7416611, 120.0910645
5: -55.9513435, 43.2402878, -61.1921082, 47.4181862, -103.3695297, 104.4323959
6: -53.4172859, 54.4014511, -58.5567780, 59.2337952, -112.6510773, 112.9582291
7: -62.3693047, 44.1060638, -67.7895889, 49.0151443, -111.3844452, 111.8956451
8: -68.8457184, 46.5383682, -75.6516953, 51.1608047, -120.0065231, 122.1900635
9: -53.4014587, 53.4184113, -58.4647141, 58.4282036, -111.8296661, 111.8831253

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3382612, upper bound: 107.3382152
time: 9.21 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3378656, upper bound: 107.3378656
time: 11.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.65 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.65
Output dim: 7, lower bound: -107.3416339, upper bound: 107.3416338
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.65
Output dim: 7, lower bound: -107.3416339, upper bound: 107.3417877
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.65
Output dim: 7, lower bound: -107.3382612, upper bound: 107.3382152
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.65
Output dim: 7, lower bound: -107.3378656, upper bound: 107.3378656

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -54.8241081, 43.0569534, -54.8241081, 43.0569534, -97.8810577, 97.8810577
1: -45.4302597, 38.5692978, -45.4302597, 38.5692978, -83.9995499, 83.9995422
2: -57.7149849, 35.3450928, -57.7149849, 35.3450928, -93.0600739, 93.0600739
3: -67.1298523, 31.9761009, -67.1298523, 31.9761009, -99.1059418, 99.1059418
4: -59.6581039, 45.5115700, -59.6581039, 45.5115700, -105.1696777, 105.1696777
5: -51.4531670, 39.7658691, -51.4531670, 39.7658691, -91.2190323, 91.2190323
6: -49.0709686, 50.1650696, -49.0709686, 50.1650696, -99.2360382, 99.2360382
7: -57.5313797, 40.3114319, -57.5313797, 40.3114319, -97.8428116, 97.8428116
8: -63.1841316, 42.7644730, -63.1841316, 42.7644730, -105.9486084, 105.9486084
9: -49.1056328, 49.1908188, -49.1056328, 49.1908188, -98.2964478, 98.2964478

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3376950, upper bound: 107.3377539
time: 8.42 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3373968, upper bound: 107.3373860
time: 11.16 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -54.8241081, 43.0569534, -59.6072922, 46.7624054, -101.5865173, 102.6642456
1: -45.4302597, 38.5692978, -49.3685532, 41.8875961, -87.3178406, 87.9378433
2: -57.7149849, 35.3450928, -62.8696785, 38.5685043, -96.2834854, 98.2147675
3: -67.1298523, 31.9761009, -72.8384476, 34.8074989, -101.9373474, 104.8145370
4: -59.6581039, 45.5115700, -64.8151855, 49.4113007, -109.0694046, 110.3267517
5: -51.4531670, 39.7658691, -55.9513435, 43.2402878, -94.6934509, 95.7172089
6: -49.0709686, 50.1650696, -53.4172859, 54.4014511, -103.4724197, 103.5823441
7: -57.5313797, 40.3114319, -62.3693047, 44.1060638, -101.6374435, 102.6807251
8: -63.1841316, 42.7644730, -68.8457184, 46.5383682, -109.7225037, 111.6101913
9: -49.1056328, 49.1908188, -53.4014587, 53.4184113, -102.5240173, 102.5922775

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3376950, upper bound: 107.3378534
time: 10.67 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3373968, upper bound: 107.3374958
time: 10.46 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -59.6072922, 46.7624054, -57.0502052, 44.7780647, -104.3853531, 103.8126068
1: -49.3685532, 41.8875961, -47.2509995, 40.1116104, -89.4801559, 89.1385956
2: -62.8696785, 38.5685043, -60.0182724, 36.7321777, -99.6018524, 98.5867767
3: -72.8384476, 34.8074989, -69.7752457, 33.2742615, -106.1127090, 104.5827179
4: -64.8151855, 49.4113007, -61.9291191, 47.3181458, -112.1333237, 111.3404083
5: -55.9513435, 43.2402878, -53.5782700, 41.3073845, -97.2587204, 96.8185577
6: -53.4172859, 54.4014511, -51.0108223, 52.1616135, -105.5788956, 105.4122772
7: -62.3693047, 44.1060638, -59.7324066, 41.8613510, -104.2306519, 103.8384705
8: -68.8457184, 46.5383682, -65.7701340, 44.4874268, -113.3331451, 112.3085022
9: -53.4014587, 53.4184113, -50.9824600, 50.9944191, -104.3958740, 104.4008636

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3378656, upper bound: 107.3378656
time: 8.01 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3378656, upper bound: 107.3378656
time: 8.90 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -58.1694565, 45.6525002, -59.1368256, 46.4186897, -104.5881500, 104.7893219
1: -48.1868248, 40.8902206, -48.9968262, 41.5279579, -89.7147751, 89.8870468
2: -61.2807159, 37.5295601, -62.1630821, 37.9624939, -99.2432022, 99.6926422
3: -71.1383972, 33.9131393, -72.3455124, 34.3730087, -105.5114059, 106.2586441
4: -63.2584381, 48.2432671, -64.2311096, 49.0268860, -112.2853241, 112.4743805
5: -54.6034317, 42.1677284, -55.5138092, 42.7763672, -97.3797989, 97.6815338
6: -52.0757751, 53.1521416, -52.8279419, 54.0826225, -106.1583862, 105.9800797
7: -60.9383202, 42.8391266, -61.9473381, 43.1985092, -104.1368256, 104.7864380
8: -67.0967712, 45.3691101, -68.0928040, 46.0525970, -113.1493683, 113.4619064
9: -52.0795784, 52.1087112, -52.7940750, 52.8444824, -104.9240570, 104.9027863

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3264091, upper bound: 107.3248596
time: 14.46 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3353310, upper bound: 107.3353311
time: 9.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.96 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.96
Output dim: 7, lower bound: -107.3376950, upper bound: 107.3377539
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.96
Output dim: 7, lower bound: -107.3373968, upper bound: 107.3373860
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.96
Output dim: 7, lower bound: -107.3376950, upper bound: 107.3378534
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.96
Output dim: 7, lower bound: -107.3373968, upper bound: 107.3374958
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.96
Output dim: 7, lower bound: -107.3378656, upper bound: 107.3378656
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.96
Output dim: 7, lower bound: -107.3378656, upper bound: 107.3378656
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.96
Output dim: 7, lower bound: -107.3264091, upper bound: 107.3248596
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.96
Output dim: 7, lower bound: -107.3353310, upper bound: 107.3353311

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -48.1095810, 37.8789062, -54.8241081, 43.0569534, -91.1665344, 92.7030182
1: -39.8699074, 33.8936691, -45.4302597, 38.5692978, -78.4392014, 79.3239288
2: -50.3132095, 30.4450417, -57.7149849, 35.3450928, -85.6583023, 88.1600189
3: -59.1242371, 27.7986488, -67.1298523, 31.9761009, -91.1003265, 94.9284973
4: -52.3907585, 40.0504532, -59.6581039, 45.5115700, -97.9023285, 99.7085495
5: -45.1852875, 34.7259750, -51.4531670, 39.7658691, -84.9511490, 86.1791382
6: -42.7712517, 44.3363609, -49.0709686, 50.1650696, -92.9363251, 93.4073257
7: -50.8120193, 34.3054428, -57.5313797, 40.3114319, -91.1234283, 91.8368225
8: -54.9759903, 37.3548355, -63.1841316, 42.7644730, -97.7404633, 100.5389633
9: -42.8681259, 43.0021667, -49.1056328, 49.1908188, -92.0589447, 92.1077957

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3376095, upper bound: 107.3376095
time: 10.31 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3376095, upper bound: 107.3376095
time: 36.49 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -50.2708092, 39.5777512, -53.4627304, 42.0082741, -92.2790756, 93.0404739
1: -41.6728363, 35.3616219, -44.3105621, 37.6248627, -79.2976990, 79.6721725
2: -52.5388908, 31.7185764, -56.2135696, 34.3545189, -86.8934097, 87.9321442
3: -61.7833481, 28.9371643, -65.5160904, 31.1283569, -92.9117050, 94.4532471
4: -54.7636414, 41.8243256, -58.1844482, 44.4098053, -99.1734467, 100.0087585
5: -47.1953812, 36.2404022, -50.1800079, 38.7479172, -85.9432983, 86.4204102
6: -44.6537018, 46.3230362, -47.8004913, 48.9857407, -93.6394424, 94.1235275
7: -53.0938339, 35.6822052, -56.1773415, 39.1024246, -92.1962585, 91.8595428
8: -57.3781776, 38.9804192, -61.5277100, 41.6639061, -99.0420837, 100.5081177
9: -44.7329330, 44.9132347, -47.8509521, 47.9447174, -92.6776505, 92.7641830

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3340196, upper bound: 107.3340574
time: 7.41 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3340334, upper bound: 107.3340334
time: 8.88 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -48.1095810, 37.8789062, -59.6072922, 46.7624054, -94.8719864, 97.4861908
1: -39.8699074, 33.8936691, -49.3685532, 41.8875961, -81.7574997, 83.2622223
2: -50.3132095, 30.4450417, -62.8696785, 38.5685043, -88.8817062, 93.3147125
3: -59.1242371, 27.7986488, -72.8384476, 34.8074989, -93.9317245, 100.6370926
4: -52.3907585, 40.0504532, -64.8151855, 49.4113007, -101.8020554, 104.8656311
5: -45.1852875, 34.7259750, -55.9513435, 43.2402878, -88.4255676, 90.6773224
6: -42.7712517, 44.3363609, -53.4172859, 54.4014511, -97.1726990, 97.7536316
7: -50.8120193, 34.3054428, -62.3693047, 44.1060638, -94.9180603, 96.6747360
8: -54.9759903, 37.3548355, -68.8457184, 46.5383682, -101.5143585, 106.2005539
9: -42.8681259, 43.0021667, -53.4014587, 53.4184113, -96.2865372, 96.4036255

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3373965, upper bound: 107.3374958
time: 11.04 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3373965, upper bound: 107.3374958
time: 11.23 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -50.2708092, 39.5777512, -58.1694565, 45.6525002, -95.9233093, 97.7472000
1: -41.6728363, 35.3616219, -48.1868248, 40.8902206, -82.5630569, 83.5484390
2: -52.5388908, 31.7185764, -61.2807159, 37.5295601, -90.0684509, 92.9992905
3: -61.7833481, 28.9371643, -71.1383972, 33.9131393, -95.6964874, 100.0755615
4: -54.7636414, 41.8243256, -63.2584381, 48.2432671, -103.0069122, 105.0827637
5: -47.1953812, 36.2404022, -54.6034317, 42.1677284, -89.3631058, 90.8438339
6: -44.6537018, 46.3230362, -52.0757751, 53.1521416, -97.8058472, 98.3988113
7: -53.0938339, 35.6822052, -60.9383202, 42.8391266, -95.9329529, 96.6205139
8: -57.3781776, 38.9804192, -67.0967712, 45.3691101, -102.7472839, 106.0771942
9: -44.7329330, 44.9132347, -52.0795784, 52.1087112, -96.8416443, 96.9928131

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3245893, upper bound: 107.3261663
time: 13.74 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3350087, upper bound: 107.3349743
time: 12.78 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -52.4242134, 41.2226181, -57.0502052, 44.7780647, -97.2022629, 98.2728119
1: -43.4426842, 36.8953171, -47.2509995, 40.1116104, -83.5542908, 84.1463165
2: -54.9425316, 33.3453331, -60.0182724, 36.7321777, -91.6746979, 93.3636017
3: -64.3209229, 30.3413277, -69.7752457, 33.2742615, -97.5951767, 100.1165466
4: -57.0419464, 43.5852089, -61.9291191, 47.3181458, -104.3600769, 105.5143127
5: -49.2427559, 37.8631516, -53.5782700, 41.3073845, -90.5501328, 91.4414215
6: -46.7045975, 48.1714783, -51.0108223, 52.1616135, -98.8662033, 99.1822968
7: -55.1919060, 37.7223396, -59.7324066, 41.8613510, -97.0532532, 97.4547424
8: -60.1002731, 40.7327309, -65.7701340, 44.4874268, -104.5876999, 106.5028687
9: -46.7566566, 46.8263206, -50.9824600, 50.9944191, -97.7510757, 97.8087769

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3347741, upper bound: 107.3347482
time: 11.02 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3348636, upper bound: 107.3348127
time: 9.35 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -54.5237923, 42.8739929, -57.0502052, 44.7780647, -99.3018494, 99.9241791
1: -45.1937828, 38.3173866, -47.2509995, 40.1116104, -85.3053741, 85.5683823
2: -57.0943184, 34.5744362, -60.0182724, 36.7321777, -93.8264847, 94.5927124
3: -66.9055710, 31.4404564, -69.7752457, 33.2742615, -100.1798325, 101.2156982
4: -59.3512115, 45.3042984, -61.9291191, 47.3181458, -106.6693573, 107.2334137
5: -51.1938515, 39.3301048, -53.5782700, 41.3073845, -92.5012360, 92.9083710
6: -48.5239334, 50.1053200, -51.0108223, 52.1616135, -100.6855392, 101.1161346
7: -57.4123383, 39.0424576, -59.7324066, 41.8613510, -99.2736893, 98.7748642
8: -62.4286461, 42.3086014, -65.7701340, 44.4874268, -106.9160767, 108.0787354
9: -48.5639076, 48.6808853, -50.9824600, 50.9944191, -99.5583267, 99.6633377

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3347741, upper bound: 107.3347482
time: 9.98 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3348636, upper bound: 107.3348127
time: 8.45 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -46.7749252, 36.8691444, -57.3362961, 45.0308418, -91.8057632, 94.2054291
1: -38.7966881, 32.9446869, -47.5256615, 40.2796669, -79.0763397, 80.4703522
2: -48.8278122, 29.3446770, -60.1928787, 36.6780243, -85.5058365, 89.5375519
3: -57.5762825, 26.7238197, -70.2223206, 33.2431946, -90.8194733, 96.9461365
4: -51.1505127, 39.0123596, -62.3198967, 47.5759506, -98.7264557, 101.3322525
5: -43.8664665, 33.7720528, -53.8232880, 41.4501610, -85.3166275, 87.5953369
6: -41.5266914, 43.2183990, -51.1729279, 52.5177078, -94.0444031, 94.3913269
7: -49.6217957, 32.9698715, -60.1658821, 41.6587410, -91.2805099, 93.1357346
8: -53.2446022, 36.2485199, -65.9135208, 44.6072540, -97.8518524, 102.1620255
9: -41.6813278, 41.9080734, -51.1642151, 51.2424850, -92.9237976, 93.0722885

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3225364, upper bound: 107.3206603
time: 11.75 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3228053, upper bound: 107.3209744
time: 12.13 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -54.1512947, 42.5591316, -59.1368256, 46.4186897, -100.5699844, 101.6959534
1: -44.8914032, 38.1131630, -48.9968262, 41.5279579, -86.4193573, 87.1099777
2: -56.8879509, 34.6644211, -62.1630821, 37.9624939, -94.8504410, 96.8274918
3: -66.3829346, 31.4193325, -72.3455124, 34.3730087, -100.7559433, 103.7648392
4: -58.9764633, 45.0101929, -64.2311096, 49.0268860, -108.0033493, 109.2413025
5: -50.8341522, 39.2122879, -55.5138092, 42.7763672, -93.6105194, 94.7260971
6: -48.3766212, 49.6610413, -52.8279419, 54.0826225, -102.4592438, 102.4889755
7: -56.9501610, 39.4053574, -61.9473381, 43.1985092, -100.1486664, 101.3526917
8: -62.2462158, 42.1601448, -68.0928040, 46.0525970, -108.2988129, 110.2529449
9: -48.4370995, 48.5219078, -52.7940750, 52.8444824, -101.2815857, 101.3159790

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3317031, upper bound: 107.3317549
time: 9.69 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3319445, upper bound: 107.3319445
time: 9.37 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 20.43 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.43
Output dim: 7, lower bound: -107.3376095, upper bound: 107.3376095
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.43
Output dim: 7, lower bound: -107.3376095, upper bound: 107.3376095
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.43
Output dim: 7, lower bound: -107.3340196, upper bound: 107.3340574
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.43
Output dim: 7, lower bound: -107.3340334, upper bound: 107.3340334
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.43
Output dim: 7, lower bound: -107.3373965, upper bound: 107.3374958
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.43
Output dim: 7, lower bound: -107.3373965, upper bound: 107.3374958
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.43
Output dim: 7, lower bound: -107.3245893, upper bound: 107.3261663
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.43
Output dim: 7, lower bound: -107.3350087, upper bound: 107.3349743
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.43
Output dim: 7, lower bound: -107.3347741, upper bound: 107.3347482
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.43
Output dim: 7, lower bound: -107.3348636, upper bound: 107.3348127
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.43
Output dim: 7, lower bound: -107.3347741, upper bound: 107.3347482
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.43
Output dim: 7, lower bound: -107.3348636, upper bound: 107.3348127
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.43
Output dim: 7, lower bound: -107.3225364, upper bound: 107.3206603
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.43
Output dim: 7, lower bound: -107.3228053, upper bound: 107.3209744
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.43
Output dim: 7, lower bound: -107.3317031, upper bound: 107.3317549
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.43
Output dim: 7, lower bound: -107.3319445, upper bound: 107.3319445

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -48.1095810, 37.8789062, -48.1095810, 37.8789062, -85.9884872, 85.9884872
1: -39.8699074, 33.8936691, -39.8699074, 33.8936691, -73.7635803, 73.7635803
2: -50.3132095, 30.4450417, -50.3132095, 30.4450417, -80.7582397, 80.7582397
3: -59.1242371, 27.7986488, -59.1242371, 27.7986488, -86.9228821, 86.9228821
4: -52.3907585, 40.0504532, -52.3907585, 40.0504532, -92.4412079, 92.4412079
5: -45.1852875, 34.7259750, -45.1852875, 34.7259750, -79.9112549, 79.9112549
6: -42.7712517, 44.3363609, -42.7712517, 44.3363609, -87.1076050, 87.1076126
7: -50.8120193, 34.3054428, -50.8120193, 34.3054428, -85.1174393, 85.1174393
8: -54.9759903, 37.3548355, -54.9759903, 37.3548355, -92.3308258, 92.3308258
9: -42.8681259, 43.0021667, -42.8681259, 43.0021667, -85.8702927, 85.8702927

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3343997, upper bound: 107.3344679
time: 9.45 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3342819, upper bound: 107.3343561
time: 10.47 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -48.1095810, 37.8789062, -50.2708092, 39.5777512, -87.6873322, 88.1497040
1: -39.8699074, 33.8936691, -41.6728363, 35.3616219, -75.2315216, 75.5665054
2: -50.3132095, 30.4450417, -52.5388908, 31.7185764, -82.0317841, 82.9839249
3: -59.1242371, 27.7986488, -61.7833481, 28.9371643, -88.0613861, 89.5820007
4: -52.3907585, 40.0504532, -54.7636414, 41.8243256, -94.2150879, 94.8140869
5: -45.1852875, 34.7259750, -47.1953812, 36.2404022, -81.4256744, 81.9213562
6: -42.7712517, 44.3363609, -44.6537018, 46.3230362, -89.0942841, 88.9900589
7: -50.8120193, 34.3054428, -53.0938339, 35.6822052, -86.4942093, 87.3992767
8: -54.9759903, 37.3548355, -57.3781776, 38.9804192, -93.9564056, 94.7330170
9: -42.8681259, 43.0021667, -44.7329330, 44.9132347, -87.7813568, 87.7350998

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3343997, upper bound: 107.3344679
time: 12.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3342819, upper bound: 107.3343561
time: 10.12 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -50.2708092, 39.5777512, -47.2387238, 37.1878319, -87.4586258, 86.8164749
1: -41.6728363, 35.3616219, -39.1361885, 33.2691078, -74.9419403, 74.4978104
2: -52.5388908, 31.7185764, -49.4091454, 29.8678932, -82.4067841, 81.1277237
3: -61.7833481, 28.9371643, -58.0530434, 27.2081127, -88.9914627, 86.9902039
4: -54.7636414, 41.8243256, -51.5228462, 39.3302917, -94.0939331, 93.3471680
5: -47.1953812, 36.2404022, -44.3228683, 34.1273651, -81.3227310, 80.5632706
6: -44.6537018, 46.3230362, -41.9865723, 43.5330315, -88.1867294, 88.3096085
7: -53.0938339, 35.6822052, -49.9103432, 33.6939316, -86.7877655, 85.5925446
8: -57.3781776, 38.9804192, -53.9501610, 36.6671143, -94.0452881, 92.9305801
9: -44.7329330, 44.9132347, -42.1190529, 42.2620888, -86.9950256, 87.0322876

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3339889, upper bound: 107.3339889
time: 7.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3339889, upper bound: 107.3340131
time: 9.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -49.5116234, 38.9893417, -48.2017479, 37.9472961, -87.4589157, 87.1910858
1: -41.0392075, 34.8293381, -39.9442978, 33.9503746, -74.9895782, 74.7736359
2: -51.7104988, 31.1717091, -50.4636726, 30.5360222, -82.2465210, 81.6353836
3: -60.8684387, 28.4574242, -59.2292862, 27.7799435, -88.6483765, 87.6867065
4: -53.9498291, 41.2009163, -52.5787086, 40.1177750, -94.0676041, 93.7796249
5: -46.4788475, 35.6753845, -45.2373428, 34.8343964, -81.3132324, 80.9127274
6: -43.9414597, 45.6564598, -42.8734093, 44.3970337, -88.3384933, 88.5298691
7: -52.3323631, 35.0193214, -50.9067612, 34.4876518, -86.8200150, 85.9260635
8: -56.4522552, 38.3723297, -55.1094284, 37.4245033, -93.8767548, 93.4817581
9: -44.0347443, 44.2175446, -42.9989662, 43.1281128, -87.1628571, 87.2165070

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3340131, upper bound: 107.3339974
time: 10.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3340131, upper bound: 107.3340334
time: 8.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -48.1095810, 37.8789062, -52.4242134, 41.2226181, -89.3321991, 90.3031158
1: -39.8699074, 33.8936691, -43.4426842, 36.8953171, -76.7652283, 77.3363495
2: -50.3132095, 30.4450417, -54.9425316, 33.3453331, -83.6585388, 85.3875504
3: -59.1242371, 27.7986488, -64.3209229, 30.3413277, -89.4655457, 92.1195679
4: -52.3907585, 40.0504532, -57.0419464, 43.5852089, -95.9759674, 97.0923843
5: -45.1852875, 34.7259750, -49.2427559, 37.8631516, -83.0484161, 83.9687347
6: -42.7712517, 44.3363609, -46.7045975, 48.1714783, -90.9427338, 91.0409393
7: -50.8120193, 34.3054428, -55.1919060, 37.7223396, -88.5343475, 89.4973373
8: -54.9759903, 37.3548355, -60.1002731, 40.7327309, -95.7087250, 97.4551086
9: -42.8681259, 43.0021667, -46.7566566, 46.8263206, -89.6944427, 89.7588196

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3343870, upper bound: 107.3344001
time: 10.05 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3343494, upper bound: 107.3343507
time: 12.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -48.1095810, 37.8789062, -54.5237923, 42.8739929, -90.9835663, 92.4026947
1: -39.8699074, 33.8936691, -45.1937828, 38.3173866, -78.1872864, 79.0874481
2: -50.3132095, 30.4450417, -57.0943184, 34.5744362, -84.8876495, 87.5393448
3: -59.1242371, 27.7986488, -66.9055710, 31.4404564, -90.5646973, 94.7042236
4: -52.3907585, 40.0504532, -59.3512115, 45.3042984, -97.6950531, 99.4016647
5: -45.1852875, 34.7259750, -51.1938515, 39.3301048, -84.5153732, 85.9198303
6: -42.7712517, 44.3363609, -48.5239334, 50.1053200, -92.8765717, 92.8602905
7: -50.8120193, 34.3054428, -57.4123383, 39.0424576, -89.8544769, 91.7177734
8: -54.9759903, 37.3548355, -62.4286461, 42.3086014, -97.2845917, 99.7834778
9: -42.8681259, 43.0021667, -48.5639076, 48.6808853, -91.5490112, 91.5660706

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3343870, upper bound: 107.3344001
time: 12.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3343494, upper bound: 107.3343507
time: 11.19 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -48.7994690, 38.4404411, -46.7749252, 36.8691444, -85.6686020, 85.2153625
1: -40.4564209, 34.3350983, -38.7966881, 32.9446869, -73.4011078, 73.1317749
2: -50.9367599, 30.6627617, -48.8278122, 29.3446770, -80.2814331, 79.4905701
3: -60.0209045, 28.0079155, -57.5762825, 26.7238197, -86.7447205, 85.5841904
4: -53.2028618, 40.6249771, -51.1505127, 39.0123596, -92.2152252, 91.7754898
5: -45.8060379, 35.1580391, -43.8664665, 33.7720528, -79.5780945, 79.0245056
6: -43.2901917, 45.0360374, -41.5266914, 43.2183990, -86.5085831, 86.5627136
7: -51.6370468, 34.4046021, -49.6217957, 32.9698715, -84.6069183, 84.0263901
8: -55.5859337, 37.8041496, -53.2446022, 36.2485199, -91.8344269, 91.0487518
9: -43.3905945, 43.5919342, -41.6813278, 41.9080734, -85.2986679, 85.2732315

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3204738, upper bound: 107.3222664
time: 10.52 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3206343, upper bound: 107.3223968
time: 8.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -50.2708092, 39.5777512, -54.1512947, 42.5591316, -92.8299332, 93.7290421
1: -41.6728363, 35.3616219, -44.8914032, 38.1131630, -79.7859955, 80.2530212
2: -52.5388908, 31.7185764, -56.8879509, 34.6644211, -87.2033081, 88.6065292
3: -61.7833481, 28.9371643, -66.3829346, 31.4193325, -93.2026825, 95.3200989
4: -54.7636414, 41.8243256, -58.9764633, 45.0101929, -99.7738342, 100.8007889
5: -47.1953812, 36.2404022, -50.8341522, 39.2122879, -86.4076691, 87.0745468
6: -44.6537018, 46.3230362, -48.3766212, 49.6610413, -94.3147430, 94.6996613
7: -53.0938339, 35.6822052, -56.9501610, 39.4053574, -92.4991913, 92.6323547
8: -57.3781776, 38.9804192, -62.2462158, 42.1601448, -99.5383224, 101.2266235
9: -44.7329330, 44.9132347, -48.4370995, 48.5219078, -93.2548370, 93.3503342

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3315079, upper bound: 107.3313433
time: 10.32 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3315215, upper bound: 107.3313786
time: 10.02 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -52.4242134, 41.2226181, -50.4067459, 39.6340790, -92.0582809, 91.6293564
1: -43.4426842, 36.8953171, -41.7439117, 35.4593582, -78.9020386, 78.6392288
2: -54.9425316, 33.3453331, -52.7445755, 31.9561806, -86.8987045, 86.0899048
3: -64.3209229, 30.3413277, -61.8438187, 29.0937195, -93.4146347, 92.1851501
4: -57.0419464, 43.5852089, -54.8123245, 41.9099960, -98.9519119, 98.3975372
5: -49.2427559, 37.8631516, -47.3329391, 36.3658333, -85.6085739, 85.1960678
6: -46.7045975, 48.1714783, -44.8156853, 46.3483467, -93.0529404, 92.9871674
7: -55.1919060, 37.7223396, -53.0555916, 36.1272888, -91.3191986, 90.7779312
8: -60.1002731, 40.7327309, -57.6974411, 39.1520309, -99.2522964, 98.4301529
9: -46.7566566, 46.8263206, -44.8989677, 44.9593964, -91.7160492, 91.7252884

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3354613, upper bound: 107.3354613
time: 9.92 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3354613, upper bound: 107.3354934
time: 10.81 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -51.6197853, 40.5996742, -51.4637337, 40.4655876, -92.0853729, 92.0634079
1: -42.7722321, 36.3326263, -42.6300049, 36.2072678, -78.9794998, 78.9626312
2: -54.0626564, 32.7681618, -53.9042358, 32.6950607, -86.7577209, 86.6723938
3: -63.3557053, 29.8376274, -63.1203651, 29.7266712, -93.0823746, 92.9579926
4: -56.1813698, 42.9271545, -55.9634972, 42.7703743, -98.9517365, 98.8906555
5: -48.4857101, 37.2651901, -48.3341293, 37.1425972, -85.6283035, 85.5993195
6: -45.9536324, 47.4664536, -45.7850037, 47.2956085, -93.2492371, 93.2514572
7: -54.3835144, 37.0280952, -54.1452866, 37.0014038, -91.3849182, 91.1733856
8: -59.1209679, 40.0880966, -58.9728699, 39.9815865, -99.1025543, 99.0609665
9: -46.0182457, 46.0932922, -45.8621178, 45.9051056, -91.9233551, 91.9554138

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3250073, upper bound: 107.3231373
time: 12.27 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3329761, upper bound: 107.3329761
time: 8.28 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -54.5237923, 42.8739929, -50.4067459, 39.6340790, -94.1578674, 93.2807312
1: -45.1937828, 38.3173866, -41.7439117, 35.4593582, -80.6531296, 80.0612946
2: -57.0943184, 34.5744362, -52.7445755, 31.9561806, -89.0504913, 87.3190155
3: -66.9055710, 31.4404564, -61.8438187, 29.0937195, -95.9992905, 93.2842712
4: -59.3512115, 45.3042984, -54.8123245, 41.9099960, -101.2612076, 100.1166229
5: -51.1938515, 39.3301048, -47.3329391, 36.3658333, -87.5596771, 86.6630249
6: -48.5239334, 50.1053200, -44.8156853, 46.3483467, -94.8722763, 94.9209976
7: -57.4123383, 39.0424576, -53.0555916, 36.1272888, -93.5396271, 92.0980530
8: -62.4286461, 42.3086014, -57.6974411, 39.1520309, -101.5806732, 100.0060272
9: -48.5639076, 48.6808853, -44.8989677, 44.9593964, -93.5233002, 93.5798492

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3346306, upper bound: 107.3345507
time: 10.11 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3346306, upper bound: 107.3347371
time: 8.95 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 20.50 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3343997, upper bound: 107.3344679
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3342819, upper bound: 107.3343561
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3343997, upper bound: 107.3344679
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3342819, upper bound: 107.3343561
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3339889, upper bound: 107.3339889
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3339889, upper bound: 107.3340131
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3340131, upper bound: 107.3339974
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3340131, upper bound: 107.3340334
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3343870, upper bound: 107.3344001
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3343494, upper bound: 107.3343507
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3343870, upper bound: 107.3344001
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3343494, upper bound: 107.3343507
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3204738, upper bound: 107.3222664
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3206343, upper bound: 107.3223968
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3315079, upper bound: 107.3313433
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3315215, upper bound: 107.3313786
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3354613, upper bound: 107.3354613
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3354613, upper bound: 107.3354934
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3250073, upper bound: 107.3231373
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3329761, upper bound: 107.3329761
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3346306, upper bound: 107.3345507
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.50
Output dim: 7, lower bound: -107.3346306, upper bound: 107.3347371
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.50
Output dim: 7, lower bound: -107.3348636, upper bound: 107.3348127
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.50
Output dim: 7, lower bound: -107.3225364, upper bound: 107.3206603
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.50
Output dim: 7, lower bound: -107.3228053, upper bound: 107.3209744
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.50
Output dim: 7, lower bound: -107.3317031, upper bound: 107.3317549
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.50
Output dim: 7, lower bound: -107.3319445, upper bound: 107.3319445
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=120.15695190429688
rel_dist={7: [-107.34558116528925, 107.34558116528925]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3417672, upper bound: 107.3417143
time: 10.84 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3420384, upper bound: 107.3420384
time: 10.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 21.84 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 21.84
Output dim: 7, lower bound: -107.3417672, upper bound: 107.3417143
IS_A2, status: Status.UNKNOWN, split count: 1, time: 21.84
Output dim: 7, lower bound: -107.3420384, upper bound: 107.3420384

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -54.8241081, 43.0569534, -62.3754387, 48.9225540, -103.7466583, 105.4323883
1: -45.4302597, 38.5692978, -51.6440125, 43.8149452, -89.2451935, 90.2133102
2: -57.7149849, 35.3450928, -65.9461365, 40.6444244, -98.3594055, 101.2912292
3: -67.1298523, 31.9761009, -76.0626755, 36.6205292, -103.7503815, 108.0387650
4: -59.6581039, 45.5115700, -67.7385406, 51.6597900, -111.3178940, 113.2501068
5: -51.4531670, 39.7658691, -58.5638390, 45.3444405, -96.7975922, 98.3297119
6: -49.0709686, 50.1650696, -56.0134315, 56.7901382, -105.8611069, 106.1784897
7: -57.5313797, 40.3114319, -65.0433731, 46.6732025, -104.2045822, 105.3547974
8: -63.1841316, 42.7644730, -72.2916870, 48.8856812, -112.0698013, 115.0561600
9: -49.1056328, 49.1908188, -55.9541321, 55.9477806, -105.0534058, 105.1449509

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3376072, upper bound: 107.3375540
time: 13.13 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3374019, upper bound: 107.3373860
time: 9.75 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -59.6072922, 46.7624054, -61.8887863, 48.5410309, -108.1483231, 108.6511841
1: -49.3685532, 41.8875961, -51.2407303, 43.4870682, -92.8555984, 93.1283264
2: -62.8696785, 38.5685043, -65.4233093, 40.3090744, -103.1787567, 103.9918137
3: -72.8384476, 34.8074989, -75.4736023, 36.3298569, -109.1682968, 110.2810898
4: -64.8151855, 49.4113007, -67.2047501, 51.2609634, -116.0761337, 116.6160507
5: -55.9513435, 43.2402878, -58.1087341, 44.9828186, -100.9341583, 101.3490219
6: -53.4172859, 54.4014511, -55.5685501, 56.3615913, -109.7788696, 109.9700012
7: -62.3693047, 44.1060638, -64.5543137, 46.2743759, -108.6436768, 108.6603622
8: -68.8457184, 46.5383682, -71.7171097, 48.4900665, -117.3357849, 118.2554779
9: -53.4014587, 53.4184113, -55.5101318, 55.5003510, -108.9018097, 108.9285355

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3380958, upper bound: 107.3380639
time: 11.81 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3378541, upper bound: 107.3378541
time: 9.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.17 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.17
Output dim: 7, lower bound: -107.3376072, upper bound: 107.3375540
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.17
Output dim: 7, lower bound: -107.3374019, upper bound: 107.3373860
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.17
Output dim: 7, lower bound: -107.3380958, upper bound: 107.3380639
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.17
Output dim: 7, lower bound: -107.3378541, upper bound: 107.3378541

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -52.6905518, 41.4118538, -54.6219101, 42.9051781, -95.5957336, 96.0337677
1: -43.6704826, 37.0869293, -45.2573166, 38.4228401, -82.0933228, 82.3442459
2: -55.3603706, 33.7887115, -57.3757133, 35.0246086, -90.3849792, 91.1644211
3: -64.5969925, 30.6501904, -66.9068832, 31.7829323, -96.3799286, 97.5570602
4: -57.3458176, 43.7822838, -59.3403587, 45.3502693, -102.6960754, 103.1226349
5: -49.4605865, 38.1654396, -51.2983627, 39.5164604, -88.9770508, 89.4638062
6: -47.0748940, 48.3138199, -48.7769814, 50.0400314, -97.1149292, 97.0907822
7: -55.3987045, 38.4087181, -57.3142471, 39.8105240, -95.2092133, 95.7229538
8: -60.5871353, 41.0398254, -62.8441887, 42.5440636, -103.1311951, 103.8840179
9: -47.1273727, 47.2282677, -48.7850418, 48.8315811, -95.9589539, 96.0133057

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3340431, upper bound: 107.3339188
time: 12.55 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3341511, upper bound: 107.3340495
time: 11.49 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -51.1851387, 40.2538605, -56.7380714, 44.5696487, -95.7547684, 96.9919205
1: -42.4329605, 36.0440750, -47.0269127, 39.8604164, -82.2933731, 83.0709839
2: -53.7044792, 32.6984406, -59.5528107, 36.2725449, -89.9770203, 92.2512512
3: -62.8117332, 29.7094460, -69.5094070, 32.8987808, -95.7105103, 99.2188568
4: -55.7282257, 42.5663605, -61.6746216, 47.0832825, -102.8115082, 104.2409668
5: -48.0562744, 37.0401459, -53.2637100, 41.0070610, -89.0633392, 90.3038483
6: -45.6705399, 47.0133133, -50.6196404, 51.9884338, -97.6589737, 97.6329422
7: -53.9044685, 37.0769081, -59.5587616, 41.1660118, -95.0704803, 96.6356659
8: -58.7438278, 39.8295174, -65.1972275, 44.1343002, -102.8781281, 105.0267334
9: -45.7381973, 45.8624039, -50.6218033, 50.7077560, -96.4459534, 96.4841919

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3338760, upper bound: 107.3337831
time: 11.58 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3339796, upper bound: 107.3339023
time: 11.37 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -57.3415871, 45.0135689, -54.1888733, 42.5705223, -99.9120941, 99.2024384
1: -47.5052605, 40.3153954, -44.8996162, 38.1322594, -85.6375198, 85.2149963
2: -60.3663826, 36.9273415, -56.9133911, 34.7306137, -95.0970001, 93.8407288
3: -70.1589890, 33.4007454, -66.3815536, 31.5263767, -101.6853638, 99.7822952
4: -62.3586578, 47.5712204, -58.8679848, 45.0000343, -107.3586884, 106.4392090
5: -53.8304977, 41.5476456, -50.8984985, 39.2000084, -93.0305023, 92.4461441
6: -51.3014030, 52.4328156, -48.3817787, 49.6599503, -100.9613495, 100.8145905
7: -60.1086197, 42.1031799, -56.8758621, 39.4640274, -99.5726471, 98.9790344
8: -66.0932922, 44.6978951, -62.3362961, 42.2042427, -108.2975311, 107.0341949
9: -51.3131485, 51.3442154, -48.3921432, 48.4364891, -99.7496338, 99.7363586

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3245440, upper bound: 107.3234832
time: 12.65 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3355507, upper bound: 107.3355118
time: 11.89 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -55.7474937, 43.7890472, -56.3005142, 44.2303543, -99.9778442, 100.0895386
1: -46.1977310, 39.2137756, -46.6636848, 39.5637703, -85.7614899, 85.8774567
2: -58.6101570, 35.7745285, -59.0832672, 35.9720421, -94.5821838, 94.8577957
3: -68.2775879, 32.4036942, -68.9785995, 32.6393661, -100.9169464, 101.3822937
4: -60.6419258, 46.2825699, -61.1959229, 46.7266998, -107.3686218, 107.4784851
5: -52.3376350, 40.3575172, -52.8594971, 40.6844521, -93.0220871, 93.2170105
6: -49.8162346, 51.0577469, -50.2185631, 51.6035995, -101.4198303, 101.2763062
7: -58.5284157, 40.6985855, -59.1125145, 40.8113823, -99.3397903, 99.8110962
8: -64.1545715, 43.4093933, -64.6853714, 43.7879753, -107.9425507, 108.0947647
9: -49.8511887, 49.8970451, -50.2203064, 50.3061028, -100.1572876, 100.1173401

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3244045, upper bound: 107.3233709
time: 12.88 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3353131, upper bound: 107.3353131
time: 9.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.73 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 7, lower bound: -107.3340431, upper bound: 107.3339188
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 7, lower bound: -107.3341511, upper bound: 107.3340495
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 7, lower bound: -107.3338760, upper bound: 107.3337831
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 7, lower bound: -107.3339796, upper bound: 107.3339023
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 7, lower bound: -107.3245440, upper bound: 107.3234832
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 7, lower bound: -107.3355507, upper bound: 107.3355118
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 7, lower bound: -107.3244045, upper bound: 107.3233709
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.73
Output dim: 7, lower bound: -107.3353131, upper bound: 107.3353131

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -46.6159401, 36.7049217, -52.1708908, 41.0076027, -87.6235428, 88.8758087
1: -38.6145554, 32.8321266, -43.2261887, 36.7082939, -75.3228455, 76.0583038
2: -48.7213326, 29.4107475, -54.6926880, 33.2620239, -81.9833527, 84.1034393
3: -57.3054276, 26.8203430, -63.9776268, 30.2414761, -87.5469055, 90.7979660
4: -50.8449326, 38.8189316, -56.7176704, 43.3556824, -94.2006073, 95.5366058
5: -43.7398224, 33.6577110, -48.9964027, 37.6973495, -81.4371719, 82.6541138
6: -41.3974876, 42.9887314, -46.4930725, 47.8987617, -89.2962494, 89.4818039
7: -49.2833710, 33.1276932, -54.8502693, 37.6943359, -86.9776917, 87.9779663
8: -53.1887207, 36.1656380, -59.8693047, 40.5771103, -93.7658310, 96.0349350
9: -41.5366592, 41.6779327, -46.5395393, 46.6058655, -88.1425171, 88.2174683

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3287268, upper bound: 107.3285513
time: 12.80 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3278434, upper bound: 107.3277289
time: 11.87 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -47.5023804, 37.4042892, -51.6503563, 40.6064796, -88.1088562, 89.0546417
1: -39.3576546, 33.4578209, -42.7931900, 36.3474274, -75.7050705, 76.2510071
2: -49.6898308, 30.0215454, -54.1263504, 32.8929901, -82.5828094, 84.1478806
3: -58.3882256, 27.3430653, -63.3555527, 29.9225883, -88.3108139, 90.6986160
4: -51.8170547, 39.5421333, -56.1591988, 42.9309959, -94.7480469, 95.7013245
5: -44.5819283, 34.3070564, -48.5108566, 37.3118668, -81.8937988, 82.8179016
6: -42.2112885, 43.7845497, -46.0122032, 47.4446373, -89.6558990, 89.7967529
7: -50.2005844, 33.8522911, -54.3314476, 37.2524986, -87.4530792, 88.1837387
8: -54.2520142, 36.8589439, -59.2416573, 40.1611862, -94.4132004, 96.1005936
9: -42.3446922, 42.4713936, -46.0631943, 46.1322517, -88.4769363, 88.5345917

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3286003, upper bound: 107.3284089
time: 13.64 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3277898, upper bound: 107.3276740
time: 10.00 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -45.2929611, 35.6820488, -54.2515411, 42.6461372, -87.9390945, 89.9335861
1: -37.5158463, 31.9077377, -44.9673004, 38.1201477, -75.6359940, 76.8750381
2: -47.2686691, 28.4479141, -56.8293953, 34.4790916, -81.7477417, 85.2773132
3: -55.7199554, 25.9844303, -66.5405350, 31.3287773, -87.0487366, 92.5249634
4: -49.4177017, 37.7403030, -59.0132408, 45.0617256, -94.4794235, 96.7535248
5: -42.5001488, 32.6657448, -50.9303436, 39.1559410, -81.6560822, 83.5960846
6: -40.1540794, 41.8406219, -48.3007050, 49.8171349, -89.9712143, 90.1413116
7: -47.9719009, 31.9340782, -57.0587463, 39.0055008, -86.9774017, 88.9928284
8: -51.5629196, 35.1039314, -62.1747055, 42.1395073, -93.7024231, 97.2786407
9: -40.3141632, 40.4633064, -48.3375969, 48.4476624, -88.7618103, 88.8009033

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3188326, upper bound: 107.3199667
time: 12.71 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3313106, upper bound: 107.3311843
time: 10.48 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -46.3019791, 36.4776077, -53.8877716, 42.3655930, -88.6675720, 90.3653717
1: -38.3613205, 32.6207962, -44.6634636, 37.8695984, -76.2309113, 77.2842560
2: -48.3730850, 29.1508617, -56.4374924, 34.2264175, -82.5994949, 85.5883255
3: -56.9504356, 26.5854568, -66.1045685, 31.1081333, -88.0585632, 92.6900177
4: -50.5228195, 38.5648804, -58.6226044, 44.7650909, -95.2879028, 97.1874695
5: -43.4570999, 33.4082642, -50.5909805, 38.8882446, -82.3453369, 83.9992218
6: -41.0852127, 42.7439613, -47.9671097, 49.4983482, -90.5835571, 90.7110748
7: -49.0118790, 32.7738113, -56.6962204, 38.7046471, -87.7165070, 89.4700317
8: -52.7787018, 35.8977814, -61.7420349, 41.8511009, -94.6297913, 97.6398163
9: -41.2377815, 41.3724594, -48.0070953, 48.1183128, -89.3560791, 89.3795471

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3191590, upper bound: 107.3202745
time: 10.91 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3313694, upper bound: 107.3312500
time: 11.36 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -46.1606331, 36.3935318, -49.1048584, 38.6495056, -84.8101349, 85.4983902
1: -38.2842941, 32.5118637, -40.7075882, 34.5882492, -72.8725433, 73.2194519
2: -48.1523399, 28.8948212, -51.3619041, 31.0899105, -79.2422485, 80.2567215
3: -56.8357162, 26.3407917, -60.3201675, 28.3458023, -85.1815033, 86.6609573
4: -50.4780846, 38.5077324, -53.4691467, 40.8820534, -91.3601379, 91.9768753
5: -43.2924728, 33.3075829, -46.1187515, 35.4534836, -78.7459564, 79.4263306
6: -40.9435768, 42.6813354, -43.6826096, 45.2269516, -86.1705170, 86.3639450
7: -49.0005569, 32.4101181, -51.8334846, 35.1003609, -84.1009216, 84.2435989
8: -52.4952087, 35.7554932, -56.1504402, 38.1371956, -90.6323853, 91.9059219
9: -41.1057587, 41.3310814, -43.7716446, 43.8936806, -84.9994354, 85.1027145

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3204749, upper bound: 107.3192582
time: 14.23 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3208176, upper bound: 107.3196198
time: 12.37 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -53.3702126, 41.9539909, -52.7984924, 41.4998474, -94.8700562, 94.7524719
1: -44.2439804, 37.5680733, -43.7559700, 37.1671143, -81.4110870, 81.3240433
2: -56.0246544, 34.0927773, -55.3936081, 33.7414703, -89.7661133, 89.4863739
3: -65.4543991, 30.9364853, -64.7302628, 30.6667442, -96.1211395, 95.6667480
4: -58.1290474, 44.3756866, -57.3881989, 43.8769493, -102.0059967, 101.7638702
5: -50.1066895, 38.6247406, -49.5969086, 38.1777191, -88.2843933, 88.2216339
6: -47.6446228, 48.9809418, -47.0989914, 48.4519844, -96.0966034, 96.0799332
7: -56.1646309, 38.7044487, -55.4927177, 38.2764130, -94.4410400, 94.1971588
8: -61.2942886, 41.5304832, -60.6540604, 41.0967369, -102.3910217, 102.1845398
9: -47.7078705, 47.7975845, -47.1305084, 47.1968307, -94.9047012, 94.9280930

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3319214, upper bound: 107.3319054
time: 10.87 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3321178, upper bound: 107.3320834
time: 9.92 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -44.9741287, 35.4797783, -51.2203751, 40.3147774, -85.2888947, 86.7001343
1: -37.3006325, 31.6806831, -42.4726562, 36.0219803, -73.3226013, 74.1533279
2: -46.8557434, 28.0346470, -53.5379486, 32.3333206, -79.1890640, 81.5725937
3: -55.4067268, 25.5913372, -62.9228973, 29.4558487, -84.8625793, 88.5142365
4: -49.1895676, 37.5376701, -55.7940903, 42.6155128, -91.8050766, 93.3317566
5: -42.1807404, 32.4150658, -48.0878601, 36.9335632, -79.1142960, 80.5029221
6: -39.8169479, 41.6545334, -45.5225639, 47.1729660, -86.9899139, 87.1770935
7: -47.8134003, 31.3334770, -54.0705528, 36.4386673, -84.2520676, 85.4040222
8: -51.0460663, 34.8065948, -58.5012436, 39.7273903, -90.7734528, 93.3078308
9: -40.0037766, 40.2383041, -45.5935860, 45.7659111, -85.7696838, 85.8318787

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3202722, upper bound: 107.3190891
time: 12.70 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3207150, upper bound: 107.3195401
time: 13.31 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -51.9406776, 40.8569107, -54.9177170, 43.1654129, -95.1060791, 95.7746277
1: -43.0644035, 36.5756569, -45.5276184, 38.6044655, -81.6688690, 82.1032562
2: -54.4530525, 33.0552216, -57.5731544, 34.9872704, -89.4403229, 90.6283722
3: -63.7565155, 30.0460663, -67.3376312, 31.7811890, -95.5377045, 97.3836975
4: -56.5918121, 43.2159615, -59.7242165, 45.6117935, -102.2035980, 102.9401779
5: -48.7742691, 37.5544357, -51.5665169, 39.6655045, -88.4397583, 89.1209564
6: -46.3084946, 47.7450943, -48.9441185, 50.4020004, -96.7104950, 96.6892090
7: -54.7437286, 37.4338913, -57.7366295, 39.6286125, -94.3723373, 95.1705017
8: -59.5409088, 40.3826065, -63.0130386, 42.6888466, -102.2297516, 103.3956451
9: -46.3841362, 46.4981079, -48.9642220, 49.0731239, -95.4572601, 95.4623260

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3316386, upper bound: 107.3316698
time: 11.96 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3319138, upper bound: 107.3319138
time: 11.05 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.40 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.40
Output dim: 7, lower bound: -107.3287268, upper bound: 107.3285513
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.40
Output dim: 7, lower bound: -107.3278434, upper bound: 107.3277289
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.40
Output dim: 7, lower bound: -107.3286003, upper bound: 107.3284089
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.40
Output dim: 7, lower bound: -107.3277898, upper bound: 107.3276740
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.40
Output dim: 7, lower bound: -107.3188326, upper bound: 107.3199667
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.40
Output dim: 7, lower bound: -107.3313106, upper bound: 107.3311843
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.40
Output dim: 7, lower bound: -107.3191590, upper bound: 107.3202745
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.40
Output dim: 7, lower bound: -107.3313694, upper bound: 107.3312500
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.40
Output dim: 7, lower bound: -107.3204749, upper bound: 107.3192582
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.40
Output dim: 7, lower bound: -107.3208176, upper bound: 107.3196198
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.40
Output dim: 7, lower bound: -107.3319214, upper bound: 107.3319054
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.40
Output dim: 7, lower bound: -107.3321178, upper bound: 107.3320834
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.40
Output dim: 7, lower bound: -107.3202722, upper bound: 107.3190891
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.40
Output dim: 7, lower bound: -107.3207150, upper bound: 107.3195401
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.40
Output dim: 7, lower bound: -107.3316386, upper bound: 107.3316698
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.40
Output dim: 7, lower bound: -107.3319138, upper bound: 107.3319138

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -45.6203308, 35.9376602, -47.8888931, 37.7126732, -83.3330002, 83.8265457
1: -37.7865295, 32.1383629, -39.6736641, 33.7277946, -71.5143127, 71.8120270
2: -47.6476021, 28.6996059, -50.0614929, 30.1871738, -77.8347778, 78.7610931
3: -56.1221962, 26.2116337, -58.8984451, 27.6229668, -83.7451553, 85.1100769
4: -49.7749634, 38.0139236, -52.1186943, 39.9006577, -89.6756210, 90.1326141
5: -42.8104286, 32.9225922, -45.0031548, 34.5286484, -77.3390808, 77.9257431
6: -40.4758759, 42.1187172, -42.5327911, 44.1655502, -84.6414261, 84.6515045
7: -48.2802048, 32.2691154, -50.5382233, 34.0046043, -82.2848053, 82.8073425
8: -51.9929466, 35.3917694, -54.7031593, 37.2383614, -89.2312927, 90.0949020
9: -40.6266632, 40.7736969, -42.6176758, 42.7304840, -83.3571472, 83.3913651

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3156451, upper bound: 107.3144044
time: 13.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3257562, upper bound: 107.3255470
time: 14.54 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -43.8356094, 34.5548592, -49.2212944, 38.7534828, -82.5890808, 83.7761536
1: -36.2961769, 30.8897820, -40.7615738, 34.6096344, -70.9058075, 71.6513519
2: -45.7187881, 27.4080582, -51.4224815, 30.8469582, -76.5657501, 78.8305359
3: -53.9911804, 25.1048717, -60.5708771, 28.2702599, -82.2614288, 85.6757507
4: -47.8565331, 36.5636215, -53.5906754, 40.9888382, -88.8453674, 90.1542892
5: -41.1336212, 31.6003227, -46.2264709, 35.4527092, -76.5863342, 77.8267899
6: -38.8172302, 40.5573616, -43.6466026, 45.3998680, -84.2170944, 84.2039642
7: -46.4826431, 30.7069283, -51.9543381, 34.6991501, -81.1817932, 82.6612549
8: -49.8308334, 33.9908867, -56.0974197, 38.2478333, -88.0786667, 90.0882950
9: -38.9889793, 39.1429977, -43.7321663, 43.8269196, -82.8158875, 82.8751678

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3148344, upper bound: 107.3136291
time: 14.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3248304, upper bound: 107.3246839
time: 11.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -46.5081635, 36.6385078, -47.4464073, 37.3719330, -83.8800964, 84.0849152
1: -38.5307884, 32.7646904, -39.3055458, 33.4208221, -71.9516068, 72.0702362
2: -48.6175003, 29.3114109, -49.5828133, 29.8746452, -78.4921417, 78.8942261
3: -57.2063522, 26.7348175, -58.3648567, 27.3506165, -84.5569611, 85.0996704
4: -50.7483788, 38.7379456, -51.6439171, 39.5368271, -90.2852020, 90.3818665
5: -43.6536598, 33.5732079, -44.5875397, 34.2002258, -77.8538818, 78.1607285
6: -41.2911072, 42.9156227, -42.1209106, 43.7780037, -85.0690994, 85.0365295
7: -49.1985855, 32.9956856, -50.0999947, 33.6258926, -82.8244781, 83.0956726
8: -53.0577469, 36.0859489, -54.1682510, 36.8844376, -89.9421844, 90.2541809
9: -41.4360428, 41.5684853, -42.2123337, 42.3261566, -83.7621994, 83.7808228

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3155547, upper bound: 107.3142840
time: 12.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3256289, upper bound: 107.3253983
time: 11.47 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -44.6834488, 35.2253876, -48.7421265, 38.3852005, -83.0686493, 83.9675064
1: -37.0072021, 31.4870186, -40.3620491, 34.2777557, -71.2849579, 71.8490677
2: -46.6429787, 27.9913597, -50.9045105, 30.5062981, -77.1492691, 78.8958740
3: -55.0298347, 25.6028900, -59.9960709, 27.9761696, -83.0060043, 85.5989532
4: -48.7877312, 37.2558022, -53.0776062, 40.5960770, -89.3837967, 90.3334045
5: -41.9400978, 32.2208862, -45.7774048, 35.0964203, -77.0365067, 77.9982910
6: -39.5954590, 41.3190536, -43.1996841, 44.9817505, -84.5772095, 84.5187378
7: -47.3602524, 31.3988838, -51.4782867, 34.2863312, -81.6465759, 82.8771667
8: -50.8460083, 34.6537514, -55.5193405, 37.8657761, -88.7117844, 90.1730957
9: -39.7621002, 39.9023056, -43.2943497, 43.3892670, -83.1513672, 83.1966553

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3147691, upper bound: 107.3135584
time: 13.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3247543, upper bound: 107.3246020
time: 11.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -41.2642212, 32.5553474, -44.2682076, 34.9310837, -76.1953049, 76.8235550
1: -34.1737709, 29.0734558, -36.7060165, 31.1171875, -65.2909546, 65.7794571
2: -42.8851967, 25.5699387, -45.9436684, 27.2888412, -70.1740265, 71.5136032
3: -50.8562698, 23.4308586, -54.5689278, 24.9825020, -75.8387756, 77.9997864
4: -45.1218987, 34.4535828, -48.3895302, 36.9346466, -82.0565338, 82.8431091
5: -38.6808701, 29.6967049, -41.4935760, 31.7806950, -70.4615631, 71.1902695
6: -36.3935471, 38.3099594, -38.9891205, 41.0872574, -77.4808044, 77.2990799
7: -43.9298897, 28.4127274, -47.1071320, 30.2544212, -74.1843109, 75.5198441
8: -46.6862030, 31.8963394, -50.0125237, 34.1670494, -80.8532333, 81.9088593
9: -36.6348724, 36.8234138, -39.1925163, 39.4668617, -76.1017227, 76.0159302

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3142094, upper bound: 107.3151615
time: 12.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3135036, upper bound: 107.3143587
time: 13.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -44.1712570, 34.8138771, -50.6310997, 39.8568726, -84.0281067, 85.4449768
1: -36.5849342, 31.1253548, -41.9716187, 35.6023254, -72.1872482, 73.0969696
2: -46.0498352, 27.6492786, -52.8880844, 31.9003735, -77.9502106, 80.5373306
3: -54.3706436, 25.2854958, -62.2082062, 29.0892220, -83.4598618, 87.4936981
4: -48.2192497, 36.8295860, -55.1462135, 42.1288948, -90.3481369, 91.9757996
5: -41.4441757, 31.8401966, -47.5328712, 36.4836922, -77.9278641, 79.3730621
6: -39.1138840, 40.8592339, -44.9510841, 46.6560898, -85.7699585, 85.8103027
7: -46.8513451, 30.9635468, -53.4532738, 35.9027748, -82.7540970, 84.4167938
8: -50.2015152, 34.2154808, -57.7707062, 39.2600327, -89.4615479, 91.9861908
9: -39.2910004, 39.4544830, -45.0376740, 45.2056503, -84.4966354, 84.4921570

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3254852, upper bound: 107.3254682
time: 11.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3246102, upper bound: 107.3245083
time: 10.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -42.2145309, 33.3074379, -44.0355072, 34.7506104, -76.9651413, 77.3429413
1: -34.9664345, 29.7508125, -36.5113106, 30.9538059, -65.9202423, 66.2621231
2: -43.9257164, 26.2247696, -45.6938133, 27.1267509, -71.0524597, 71.9185715
3: -52.0219650, 23.9934521, -54.2858582, 24.8388119, -76.8607788, 78.2793045
4: -46.1625977, 35.2333031, -48.1398964, 36.7410622, -82.9036407, 83.3731995
5: -39.5854034, 30.3977013, -41.2742004, 31.6072807, -71.1926880, 71.6719055
6: -37.2749252, 39.1625404, -38.7717934, 40.8819046, -78.1568298, 77.9343338
7: -44.9156036, 29.2020988, -46.8711662, 30.0577469, -74.9733505, 76.0732651
8: -47.8239403, 32.6439590, -49.7375717, 33.9833794, -81.8073196, 82.3815308
9: -37.5037766, 37.6846848, -38.9788170, 39.2508965, -76.7546692, 76.6634979

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3141910, upper bound: 107.3151925
time: 10.83 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3134174, upper bound: 107.3143058
time: 13.08 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -45.1594009, 35.5937881, -50.2939987, 39.5972710, -84.7566681, 85.8877716
1: -37.4131851, 31.8236885, -41.6904182, 35.3692703, -72.7824554, 73.5140991
2: -47.1299286, 28.3355637, -52.5265350, 31.6671524, -78.7970657, 80.8620987
3: -55.5788460, 25.8713551, -61.8039856, 28.8816242, -84.4604721, 87.6753387
4: -49.3031006, 37.6371765, -54.7855110, 41.8518944, -91.1549835, 92.4226837
5: -42.3820381, 32.5663757, -47.2177277, 36.2351837, -78.6172180, 79.7840958
6: -40.0252457, 41.7448044, -44.6415138, 46.3593330, -86.3845825, 86.3863220
7: -47.8716965, 31.7838001, -53.1174355, 35.6234016, -83.4951019, 84.9012222
8: -51.3909912, 34.9912605, -57.3696213, 38.9937325, -90.3847122, 92.3608856
9: -40.1945877, 40.3445625, -44.7296333, 44.8998871, -85.0944748, 85.0741959

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3254429, upper bound: 107.3254529
time: 13.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3244903, upper bound: 107.3243879
time: 12.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -44.2100868, 34.8790855, -43.5694046, 34.3371925, -78.5472794, 78.4484863
1: -36.6558495, 31.1339684, -36.0802307, 30.6806717, -67.3365173, 67.2142029
2: -46.0291290, 27.4962120, -45.3215637, 27.0945072, -73.1236343, 72.8177719
3: -54.4759636, 25.1001663, -53.6301346, 24.8233490, -79.2993011, 78.7303009
4: -48.3792496, 36.9062347, -47.5305099, 36.3311844, -84.7104263, 84.4367447
5: -41.4497375, 31.8544807, -40.8757591, 31.3314266, -72.7811584, 72.7302322
6: -39.0992928, 40.9681129, -38.4752312, 40.3465424, -79.4458313, 79.4433441
7: -47.0177078, 30.6946144, -46.2422295, 30.2409172, -77.2586136, 76.9368439
8: -50.1374207, 34.1965866, -49.3941460, 33.7051888, -83.8425980, 83.5907288
9: -39.3073349, 39.5314713, -38.6641655, 38.8202286, -78.1275635, 78.1956329

Time for backsubstitution: 1.34 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=120.15695190429688
rel_dist={7: [-107.34546615888509, 107.34546613728932]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3416014, upper bound: 107.3415665
time: 10.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3419960, upper bound: 107.3419960
time: 12.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 23.29 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 23.29
Output dim: 7, lower bound: -107.3416014, upper bound: 107.3415665
IS_A2, status: Status.UNKNOWN, split count: 1, time: 23.29
Output dim: 7, lower bound: -107.3419960, upper bound: 107.3419960

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -54.8241081, 43.0569534, -57.6572418, 45.2482681, -100.0723724, 100.7141953
1: -45.4302597, 38.5692978, -47.7615852, 40.5315514, -85.9617920, 86.3308868
2: -57.7149849, 35.3450928, -60.7949066, 37.3341331, -95.0491180, 96.1399994
3: -67.1298523, 31.9761009, -70.4845047, 33.7119713, -100.8418274, 102.4605942
4: -59.6581039, 45.5115700, -62.6931343, 47.8130722, -107.4711761, 108.2047043
5: -51.4531670, 39.7658691, -54.1169357, 41.8490067, -93.3021545, 93.8827972
6: -49.0709686, 50.1650696, -51.6724243, 52.6480026, -101.7189713, 101.8374863
7: -57.5313797, 40.3114319, -60.3501129, 42.6962433, -100.2276230, 100.6615295
8: -63.1841316, 42.7644730, -66.5913849, 45.0491638, -108.2332916, 109.3558578
9: -49.1056328, 49.1908188, -51.6729164, 51.7213440, -100.8269653, 100.8637390

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3373205, upper bound: 107.3373005
time: 11.08 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3372490, upper bound: 107.3372388
time: 11.42 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -59.6072922, 46.7624054, -58.4527321, 45.8627739, -105.4700546, 105.2151184
1: -49.3685532, 41.8875961, -48.4132576, 41.1021233, -90.4706726, 90.3008575
2: -62.8696785, 38.5685043, -61.6754570, 37.9037781, -100.7734528, 100.2439575
3: -72.8384476, 34.8074989, -71.4008255, 34.2184067, -107.0568390, 106.2083206
4: -64.8151855, 49.4113007, -63.5209541, 48.4569016, -113.2720795, 112.9322510
5: -55.9513435, 43.2402878, -54.8752861, 42.4382515, -98.3895874, 98.1155701
6: -53.4172859, 54.4014511, -52.4076843, 53.3424454, -106.7597275, 106.8091354
7: -62.3693047, 44.1060638, -61.1321487, 43.3882294, -105.7575226, 105.2382126
8: -68.8457184, 46.5383682, -67.5751877, 45.6924286, -114.5381470, 114.1135559
9: -53.4014587, 53.4184113, -52.3902588, 52.4145889, -105.8160477, 105.8086700

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3378854, upper bound: 107.3378724
time: 11.71 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3377898, upper bound: 107.3377898
time: 12.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.77 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.77
Output dim: 7, lower bound: -107.3373205, upper bound: 107.3373005
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.77
Output dim: 7, lower bound: -107.3372490, upper bound: 107.3372388
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.77
Output dim: 7, lower bound: -107.3378854, upper bound: 107.3378724
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.77
Output dim: 7, lower bound: -107.3377898, upper bound: 107.3377898

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -49.8236160, 39.2037315, -50.5161514, 39.7372284, -89.5608215, 89.7198792
1: -41.2980347, 35.0940132, -41.8660393, 35.5690460, -76.8670807, 76.9600525
2: -52.2031784, 31.6995754, -52.9193497, 32.1369209, -84.3401031, 84.6189270
3: -61.1811867, 28.8709526, -62.0108719, 29.2735214, -90.4547119, 90.8818207
4: -54.2485085, 41.4539642, -54.9602585, 42.0169029, -96.2653961, 96.4142227
5: -46.7912865, 36.0158691, -47.4486504, 36.4975052, -83.2887726, 83.4645004
6: -44.3902779, 45.8290405, -44.9961929, 46.4497604, -90.8400421, 90.8252335
7: -52.5271225, 35.8488197, -53.2153244, 36.3427734, -88.8698959, 89.0641479
8: -57.0855293, 38.7346611, -57.8914948, 39.2700081, -96.3555374, 96.6261597
9: -44.4612427, 44.5913086, -45.0568085, 45.1631432, -89.6243896, 89.6481171

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3336797, upper bound: 107.3336379
time: 14.20 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3338161, upper bound: 107.3337706
time: 12.11 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -48.7278099, 38.3624191, -52.6626205, 41.4250641, -90.1528625, 91.0250397
1: -40.3935547, 34.3336601, -43.6597481, 37.0265923, -77.4201431, 77.9934082
2: -51.0037079, 30.9076233, -55.1288452, 33.4014473, -84.4051514, 86.0364685
3: -59.8724556, 28.1775303, -64.6529312, 30.4026203, -90.2750778, 92.8304596
4: -53.0755844, 40.5641975, -57.3222198, 43.7779732, -96.8535614, 97.8864136
5: -45.7617188, 35.1986465, -49.4441376, 38.0038376, -83.7655563, 84.6427841
6: -43.3649521, 44.8833961, -46.8658142, 48.4239845, -91.7889404, 91.7492065
7: -51.4510193, 34.8816261, -55.4866562, 37.7107735, -89.1617889, 90.3682709
8: -55.7278900, 37.8538437, -60.2763824, 40.8863068, -96.6141968, 98.1302185
9: -43.4602470, 43.6051254, -46.9132957, 47.0625153, -90.5227661, 90.5184174

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3336237, upper bound: 107.3335907
time: 12.72 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3337527, upper bound: 107.3337168
time: 10.99 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -54.2893944, 42.6592941, -51.2005730, 40.2627907, -94.5521774, 93.8598633
1: -44.9889565, 38.1955490, -42.4289398, 36.0570564, -81.0460129, 80.6244888
2: -56.9992676, 34.7054329, -53.6725082, 32.6323471, -89.6316147, 88.3779449
3: -66.5454865, 31.5002632, -62.8137360, 29.7052994, -96.2507858, 94.3139954
4: -59.0577240, 45.0992508, -55.6719971, 42.5732613, -101.6309814, 100.7712326
5: -50.9804535, 39.2595444, -48.0997581, 37.0049057, -87.9853592, 87.3592911
6: -48.4514122, 49.7877541, -45.6294098, 47.0466385, -95.4980469, 95.4171600
7: -57.0594978, 39.3874512, -53.8895645, 36.9501343, -94.0096283, 93.2769928
8: -62.3803291, 42.2332535, -58.7418289, 39.8221283, -102.2024536, 100.9750824
9: -48.4866982, 48.5408745, -45.6821213, 45.7594795, -94.2461777, 94.2229919

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3342069, upper bound: 107.3341827
time: 12.52 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344815, upper bound: 107.3344708
time: 13.13 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -53.0765343, 41.7296219, -53.3291016, 41.9362488, -95.0127716, 95.0587234
1: -43.9944458, 37.3552933, -44.2053261, 37.4998779, -81.4943237, 81.5606232
2: -55.6652832, 33.8303871, -55.8594742, 33.8834953, -89.5487671, 89.6898651
3: -65.1071320, 30.7385902, -65.4321442, 30.8246574, -95.9317856, 96.1707230
4: -57.7610168, 44.1208878, -58.0153198, 44.3150101, -102.0760269, 102.1362076
5: -49.8478012, 38.3558044, -50.0780373, 38.4961472, -88.3439484, 88.4338379
6: -47.3219223, 48.7451324, -47.4797058, 49.0038948, -96.3258209, 96.2248383
7: -55.8637390, 38.3214264, -56.1390343, 38.3028526, -94.1665955, 94.4604568
8: -60.8935966, 41.2590523, -61.1083374, 41.4211884, -102.3147888, 102.3673782
9: -47.3771172, 47.4562950, -47.5208549, 47.6413116, -95.0184326, 94.9771423

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3341112, upper bound: 107.3340995
time: 10.41 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3344020, upper bound: 107.3344024
time: 11.85 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.68 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.68
Output dim: 7, lower bound: -107.3336797, upper bound: 107.3336379
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.68
Output dim: 7, lower bound: -107.3338161, upper bound: 107.3337706
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.68
Output dim: 7, lower bound: -107.3336237, upper bound: 107.3335907
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.68
Output dim: 7, lower bound: -107.3337527, upper bound: 107.3337168
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.68
Output dim: 7, lower bound: -107.3342069, upper bound: 107.3341827
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.68
Output dim: 7, lower bound: -107.3344815, upper bound: 107.3344708
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.68
Output dim: 7, lower bound: -107.3341112, upper bound: 107.3340995
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.68
Output dim: 7, lower bound: -107.3344020, upper bound: 107.3344024

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -44.2063828, 34.8340225, -46.1783676, 36.3711929, -80.5775757, 81.0123825
1: -36.6029587, 31.1412106, -38.2473679, 32.5206985, -69.1236496, 69.3885727
2: -46.0666008, 27.6472645, -48.1819496, 29.0091438, -75.0757446, 75.8292160
3: -54.4069481, 25.3098717, -56.7844887, 26.5285969, -80.9355392, 82.0943604
4: -48.2264404, 36.8462524, -50.3131638, 38.4597321, -86.6861725, 87.1594086
5: -41.4807549, 31.8422394, -43.3522377, 33.2718468, -74.7526016, 75.1944656
6: -39.1207161, 40.8849220, -40.9290962, 42.6352386, -81.7559357, 81.8140182
7: -46.8689003, 30.9381447, -48.8543167, 32.5621262, -79.4310226, 79.7924576
8: -50.2317963, 34.2303696, -52.5927048, 35.7937622, -86.0255585, 86.8230743
9: -39.2925491, 39.4338264, -41.0690918, 41.1906281, -80.4831772, 80.5029144

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3277160, upper bound: 107.3276493
time: 13.08 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3273778, upper bound: 107.3273276
time: 11.82 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -45.0316849, 35.4884338, -45.8237991, 36.0980606, -81.1297455, 81.3122330
1: -37.2967834, 31.7237244, -37.9529114, 32.2764893, -69.5732727, 69.6766281
2: -46.9683762, 28.2152348, -47.8018150, 28.7646503, -75.7330246, 76.0170441
3: -55.4192238, 25.7938004, -56.3602295, 26.3145237, -81.7337494, 82.1540298
4: -49.1344757, 37.5195847, -49.9336166, 38.1662216, -87.3006973, 87.4532013
5: -42.2677231, 32.4462662, -43.0214653, 33.0107727, -75.2784958, 75.4677124
6: -39.8792534, 41.6277084, -40.6037292, 42.3228531, -82.2021027, 82.2314377
7: -47.7261429, 31.6114464, -48.5051842, 32.2675056, -79.9936523, 80.1166306
8: -51.2213364, 34.8740540, -52.1708679, 35.5130310, -86.7343674, 87.0449066
9: -40.0441818, 40.1723289, -40.7450333, 40.8669090, -80.9110718, 80.9173584

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 43

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3276596, upper bound: 107.3275870
time: 12.63 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3273470, upper bound: 107.3273039
time: 9.90 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -43.1887436, 34.0437775, -48.2770195, 38.0225296, -81.2112732, 82.3207855
1: -35.7588615, 30.4268589, -40.0008392, 33.9449348, -69.7037811, 70.4276962
2: -44.9541435, 26.9072037, -50.3381348, 30.2340469, -75.1881866, 77.2453308
3: -53.1750946, 24.6599331, -59.3718452, 27.6237373, -80.7988281, 84.0317612
4: -47.1277275, 36.0172462, -52.6211090, 40.1836319, -87.3113556, 88.6383514
5: -40.5193520, 31.0821667, -45.3055344, 34.7385406, -75.2578888, 76.3877029
6: -38.1626167, 40.0021172, -42.7510300, 44.5694733, -82.7320862, 82.7531281
7: -45.8565407, 30.0175648, -51.0755768, 33.8726921, -79.7292328, 81.0931396
8: -48.9730682, 33.4151306, -54.9181595, 37.3683548, -86.3414154, 88.3332901
9: -38.3569298, 38.5078316, -42.8717499, 43.0443649, -81.4012909, 81.3795624

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3276901, upper bound: 107.3276227
time: 11.46 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3273123, upper bound: 107.3272759
time: 12.75 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -44.1921120, 34.8397369, -48.1575851, 37.9340172, -82.1261292, 82.9973145
1: -36.6016579, 31.1400375, -39.9017143, 33.8690796, -70.4707336, 71.0417480
2: -46.0533943, 27.6102448, -50.2186890, 30.1655903, -76.2189789, 77.8289337
3: -54.4037628, 25.2573338, -59.2285042, 27.5609264, -81.9646912, 84.4858398
4: -48.2275543, 36.8419685, -52.4953499, 40.0856819, -88.3132324, 89.3373184
5: -41.4749374, 31.8214073, -45.1969147, 34.6556320, -76.1305542, 77.0183258
6: -39.0930061, 40.9038658, -42.6508865, 44.4641190, -83.5571289, 83.5547485
7: -46.8946991, 30.8597355, -50.9626770, 33.7963867, -80.6910782, 81.8224106
8: -50.1824799, 34.2073593, -54.7900887, 37.2813339, -87.4638062, 88.9974289
9: -39.2763443, 39.4177628, -42.7695503, 42.9423218, -82.2186584, 82.1873093

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3276240, upper bound: 107.3275565
time: 12.33 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3272721, upper bound: 107.3272324
time: 11.79 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -48.1069603, 37.8679619, -46.7085457, 36.7766190, -84.8835754, 84.5764999
1: -39.8402367, 33.8554878, -38.6849136, 32.8969498, -72.7371826, 72.5403900
2: -50.2325211, 30.2483845, -48.7597275, 29.3907299, -79.6232529, 79.0081100
3: -59.1242714, 27.5947971, -57.4048538, 26.8611794, -85.9854507, 84.9996490
4: -52.4413605, 40.0457573, -50.8589554, 38.8913422, -91.3327026, 90.9047089
5: -45.1569481, 34.6612129, -43.8582115, 33.6568336, -78.8137589, 78.5194092
6: -42.6679649, 44.3635330, -41.4136963, 43.0961266, -85.7640839, 85.7772217
7: -50.8439407, 34.0030479, -49.3713303, 33.0353317, -83.8792725, 83.3743591
8: -54.8356819, 37.2751999, -53.2445564, 36.2206039, -91.0562897, 90.5197601
9: -42.7999039, 42.8949280, -41.5497627, 41.6506119, -84.4505157, 84.4446869

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3277160, upper bound: 107.3284544
time: 9.82 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3281770, upper bound: 107.3281638
time: 14.14 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -49.3241959, 38.8254585, -46.5533295, 36.6596489, -85.9838409, 85.3787842
1: -40.8569412, 34.7179375, -38.5558548, 32.7969170, -73.6538544, 73.2737885
2: -51.5681152, 31.1076698, -48.6009636, 29.2948742, -80.8629913, 79.7086258
3: -60.5955353, 28.3330822, -57.2164154, 26.7776604, -87.3731995, 85.5494995
4: -53.7594261, 41.0426521, -50.6913986, 38.7635040, -92.5229111, 91.7340393
5: -46.3119698, 35.5571747, -43.7149506, 33.5455093, -79.8574677, 79.2721100
6: -43.7907829, 45.4505959, -41.2757950, 42.9610291, -86.7518082, 86.7263794
7: -52.0910683, 35.0282249, -49.2237473, 32.9211731, -85.0122375, 84.2519684
8: -56.3058243, 38.2358856, -53.0684738, 36.1024818, -92.4083099, 91.3043442
9: -43.9151230, 43.9924202, -41.4131508, 41.5118828, -85.4270020, 85.4055710

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3284463, upper bound: 107.3284195
time: 13.38 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3281612, upper bound: 107.3281517
time: 10.88 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -46.9811821, 37.0015373, -48.7798462, 38.4071465, -85.3883286, 85.7813721
1: -38.9121933, 33.0717659, -40.4142113, 34.2990112, -73.2112045, 73.4859772
2: -49.0036354, 29.4334488, -50.8821526, 30.5929642, -79.5966034, 80.3155975
3: -57.7747879, 26.8779564, -59.9563751, 27.9391041, -85.7138901, 86.8343353
4: -51.2353745, 39.1329041, -53.1371574, 40.5909805, -91.8263550, 92.2700577
5: -44.0996132, 33.8211403, -45.7859497, 35.1005402, -79.2001343, 79.6070862
6: -41.6128693, 43.3890533, -43.2072487, 45.0063019, -86.6191711, 86.5962982
7: -49.7348862, 32.9948235, -51.5647812, 34.3166428, -84.0515289, 84.5596008
8: -53.4506683, 36.3766289, -55.5374680, 37.7724457, -91.2231140, 91.9140930
9: -41.7663612, 41.8782425, -43.3252144, 43.4780159, -85.2443771, 85.2034454

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3276912, upper bound: 107.3284127
time: 14.67 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3280962, upper bound: 107.3280931
time: 10.97 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -48.4639816, 38.1667557, -48.9054260, 38.5087700, -86.9727478, 87.0721817
1: -40.1504517, 34.1238632, -40.5171394, 34.3988800, -74.5493317, 74.6409988
2: -50.6317749, 30.4927235, -51.0336952, 30.7094212, -81.3411865, 81.5264053
3: -59.5706940, 27.7865734, -60.1059914, 28.0363846, -87.6070709, 87.8925629
4: -52.8415565, 40.3485184, -53.2703934, 40.6946449, -93.5362015, 93.6188889
5: -45.5072403, 34.9181824, -45.9090958, 35.2010040, -80.7082443, 80.8272781
6: -42.9907761, 44.7102394, -43.3376236, 45.1161880, -88.1069641, 88.0478668
7: -51.2502708, 34.2692451, -51.6951599, 34.4669342, -85.7171936, 85.9644012
8: -55.2539482, 37.5530701, -55.7119370, 37.8837204, -93.1376648, 93.2650070
9: -43.1308289, 43.2272301, -43.4518852, 43.6025543, -86.7333832, 86.6791153

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3283995, upper bound: 107.3283750
time: 14.29 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3280686, upper bound: 107.3280680
time: 11.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.96 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 7, lower bound: -107.3277160, upper bound: 107.3276493
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 7, lower bound: -107.3273778, upper bound: 107.3273276
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 7, lower bound: -107.3276596, upper bound: 107.3275870
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 7, lower bound: -107.3273470, upper bound: 107.3273039
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 7, lower bound: -107.3276901, upper bound: 107.3276227
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 7, lower bound: -107.3273123, upper bound: 107.3272759
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 7, lower bound: -107.3276240, upper bound: 107.3275565
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 7, lower bound: -107.3272721, upper bound: 107.3272324
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 7, lower bound: -107.3277160, upper bound: 107.3284544
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 7, lower bound: -107.3281770, upper bound: 107.3281638
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 7, lower bound: -107.3284463, upper bound: 107.3284195
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 7, lower bound: -107.3281612, upper bound: 107.3281517
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 7, lower bound: -107.3276912, upper bound: 107.3284127
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 7, lower bound: -107.3280962, upper bound: 107.3280931
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 7, lower bound: -107.3283995, upper bound: 107.3283750
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.96
Output dim: 7, lower bound: -107.3280686, upper bound: 107.3280680

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -41.5625229, 32.7832146, -42.3692627, 33.4176483, -74.9801636, 75.1524811
1: -34.3985100, 29.2866440, -35.0746193, 29.8514633, -64.2499695, 64.3612671
2: -43.2192230, 25.7535286, -44.0779953, 26.2725830, -69.4918060, 69.8315277
3: -51.2408676, 23.6956615, -52.2321434, 24.1942825, -75.4351501, 75.9278030
4: -45.3743210, 34.6949501, -46.2080727, 35.3686523, -80.7429733, 80.9030228
5: -39.0002670, 29.8849411, -39.7792091, 30.4522285, -69.4524994, 69.6641541
6: -36.6597748, 38.5709000, -37.3825417, 39.2999268, -75.9596939, 75.9534378
7: -44.1790848, 28.6391735, -44.9897499, 29.2493038, -73.4283905, 73.6289215
8: -47.0534248, 32.1738701, -48.0074272, 32.8310738, -79.8844910, 80.1812973
9: -36.8677216, 37.0191917, -37.5741730, 37.7157478, -74.5834656, 74.5933609

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3259226, upper bound: 107.3258292
time: 12.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3259448, upper bound: 107.3258549
time: 13.15 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -39.9558296, 31.5467854, -43.8281326, 34.5710945, -74.5269241, 75.3749084
1: -33.0660896, 28.1474075, -36.2642250, 30.8233032, -63.8893929, 64.4116364
2: -41.4816895, 24.5877609, -45.5807610, 27.0279770, -68.5096664, 70.1685104
3: -49.3231277, 22.6898174, -54.0552406, 24.9331169, -74.2562408, 76.7450562
4: -43.6445923, 33.3806686, -47.8003922, 36.5598183, -80.2043915, 81.1810532
5: -37.4920578, 28.6870155, -41.1353035, 31.4685211, -68.9605713, 69.8223114
6: -35.1556053, 37.1622391, -38.6088905, 40.6562729, -75.8118744, 75.7711334
7: -42.5555840, 27.2067013, -46.5267639, 30.0542774, -72.6098557, 73.7334595
8: -45.1058731, 30.9018669, -49.5736618, 33.9309578, -79.0368347, 80.4755249
9: -35.3855286, 35.5482635, -38.7965965, 38.9306717, -74.3162003, 74.3448639

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3256570, upper bound: 107.3255878
time: 13.48 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3256500, upper bound: 107.3255833
time: 8.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -42.3756714, 33.4285278, -42.1073837, 33.2173042, -75.5929718, 75.5359116
1: -35.0791245, 29.8600388, -34.8552322, 29.6697617, -64.7488861, 64.7152634
2: -44.1057968, 26.3095951, -43.7974319, 26.0926991, -70.1984940, 70.1070251
3: -52.2366524, 24.1699924, -51.9155579, 24.0365963, -76.2732468, 76.0855484
4: -46.2667046, 35.3601189, -45.9275856, 35.1473427, -81.4140320, 81.2877045
5: -39.7754822, 30.4810848, -39.5337715, 30.2592945, -70.0347672, 70.0148544
6: -37.4053802, 39.3015251, -37.1406021, 39.0694618, -76.4748383, 76.4421234
7: -45.0221405, 29.3015499, -44.7301750, 29.0293350, -74.0514603, 74.0317154
8: -48.0266876, 32.8078041, -47.6975555, 32.6233864, -80.6500702, 80.5053406
9: -37.6086845, 37.7451134, -37.3334427, 37.4767838, -75.0854568, 75.0785522

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 186

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3258606, upper bound: 107.3257684
time: 14.28 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3258820, upper bound: 107.3257950
time: 11.40 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -40.6699486, 32.1093292, -43.4229393, 34.2559586, -74.9259033, 75.5322723
1: -33.6561432, 28.6497517, -35.9226608, 30.5381031, -64.1942444, 64.5724106
2: -42.2533340, 25.0655766, -45.1309662, 26.7405987, -68.9939346, 70.1965408
3: -50.1898308, 23.1006451, -53.5667343, 24.6855659, -74.8753967, 76.6673813
4: -44.4223137, 33.9644623, -47.3622398, 36.2233086, -80.6456223, 81.3266983
5: -38.1676292, 29.2080441, -40.7489967, 31.1634579, -69.3310852, 69.9570389
6: -35.8055000, 37.8023987, -38.2275505, 40.2970047, -76.1025009, 76.0299530
7: -43.2925797, 27.7767353, -46.1163559, 29.6946220, -72.9871979, 73.8930893
8: -45.9478302, 31.4569912, -49.0810051, 33.6060486, -79.5538635, 80.5379944
9: -36.0307045, 36.1797180, -38.4230003, 38.5578690, -74.5885773, 74.6027145

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3256260, upper bound: 107.3255624
time: 10.30 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3256235, upper bound: 107.3255593
time: 11.24 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -40.6938553, 32.1136589, -44.4406738, 35.0492477, -75.7431030, 76.5543289
1: -33.6845665, 28.6733875, -36.8050079, 31.2602921, -64.9448547, 65.4783936
2: -42.2695351, 25.1221409, -46.2061691, 27.4817085, -69.7512360, 71.3283005
3: -50.1948624, 23.1380177, -54.7846489, 25.2768860, -75.4717407, 77.9226685
4: -44.4430580, 33.9851379, -48.4838524, 37.0734215, -81.5164795, 82.4689941
5: -38.1815186, 29.2359467, -41.7106590, 31.9007416, -70.0822601, 70.9466095
6: -35.8408966, 37.8181534, -39.1823044, 41.2087555, -77.0496521, 77.0004578
7: -43.3146973, 27.8456726, -47.1824913, 30.5448246, -73.8595200, 75.0281677
8: -45.9882202, 31.4795876, -50.3041344, 34.3870354, -80.3752594, 81.7837219
9: -36.0673332, 36.2346420, -39.3567619, 39.5453148, -75.6126480, 75.5914001

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3259019, upper bound: 107.3258114
time: 13.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3259178, upper bound: 107.3258335
time: 12.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -39.1283798, 30.9128418, -45.6665993, 36.0248184, -75.1531982, 76.5794373
1: -32.3859406, 27.5635815, -37.8036690, 32.0703964, -64.4563370, 65.3672485
2: -40.5779800, 23.9929428, -47.4607697, 28.0840530, -68.6620331, 71.4536972
3: -48.3340759, 22.1610069, -56.3368301, 25.8775539, -74.2116318, 78.4978333
4: -42.7569466, 32.7065353, -49.8272705, 38.0716400, -80.8285828, 82.5338058
5: -36.7178497, 28.0645466, -42.8546371, 32.7448387, -69.4626694, 70.9191818
6: -34.3762779, 36.4413223, -40.1972580, 42.3592148, -76.7354889, 76.6385727
7: -41.7355766, 26.4474182, -48.4830360, 31.1580963, -72.8936615, 74.9304504
8: -44.0986824, 30.2391491, -51.6000938, 35.3073196, -79.4060059, 81.8392410
9: -34.6210251, 34.7982979, -40.3698769, 40.5443115, -75.1653366, 75.1681747

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3255889, upper bound: 107.3255407
time: 13.42 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3255797, upper bound: 107.3255297
time: 11.37 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -41.6457367, 32.8642044, -44.3744125, 35.0019836, -76.6477203, 77.2386169
1: -34.4769897, 29.3489285, -36.7492638, 31.2200508, -65.6970367, 66.0981903
2: -43.3076286, 25.7828445, -46.1445656, 27.4497681, -70.7574005, 71.9274063
3: -51.3529930, 23.6996593, -54.7042885, 25.2445393, -76.5975189, 78.4039459
4: -45.4788094, 34.7680435, -48.4148483, 37.0173225, -82.4961319, 83.1828766
5: -39.0859642, 29.9352169, -41.6509628, 31.8567829, -70.9427490, 71.5861816
6: -36.7189789, 38.6723557, -39.1298904, 41.1504517, -77.8694229, 77.8022461
7: -44.2968445, 28.6397362, -47.1208344, 30.5117626, -74.8086090, 75.7605743
8: -47.1235657, 32.2268639, -50.2391281, 34.3418312, -81.4653854, 82.4659882
9: -36.9372330, 37.0899353, -39.3030167, 39.4900742, -76.4273071, 76.3929520

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3258364, upper bound: 107.3257436
time: 9.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3258553, upper bound: 107.3257675
time: 12.22 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -39.9886551, 31.5861473, -45.4663086, 35.8681755, -75.8568268, 77.0524597
1: -33.0966530, 28.1707954, -37.6311989, 31.9313068, -65.0279617, 65.8019943
2: -41.5083313, 24.5776062, -47.2382355, 27.9500751, -69.4584045, 71.8158417
3: -49.3738022, 22.6629314, -56.0902519, 25.7602005, -75.1340027, 78.7531815
4: -43.6891022, 33.4125977, -49.6068878, 37.9065094, -81.5956116, 83.0194778
5: -37.5291176, 28.6941433, -42.6604271, 32.5949402, -70.1240540, 71.3545609
6: -35.1674271, 37.2109604, -40.0141602, 42.1788750, -77.3462982, 77.2250977
7: -42.6176720, 27.1533279, -48.2774582, 30.9940834, -73.6117477, 75.4307861
8: -45.1155510, 30.9139023, -51.3623505, 35.1516266, -80.2671814, 82.2762527
9: -35.4038391, 35.5660477, -40.1895676, 40.3602142, -75.7640533, 75.7556152

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3255459, upper bound: 107.3254933
time: 12.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3255391, upper bound: 107.3254857
time: 12.18 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.31 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 7, lower bound: -107.3259226, upper bound: 107.3258292
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 7, lower bound: -107.3259448, upper bound: 107.3258549
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 7, lower bound: -107.3256570, upper bound: 107.3255878
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 7, lower bound: -107.3256500, upper bound: 107.3255833
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 7, lower bound: -107.3258606, upper bound: 107.3257684
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 7, lower bound: -107.3258820, upper bound: 107.3257950
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 7, lower bound: -107.3256260, upper bound: 107.3255624
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 7, lower bound: -107.3256235, upper bound: 107.3255593
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 7, lower bound: -107.3259019, upper bound: 107.3258114
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 7, lower bound: -107.3259178, upper bound: 107.3258335
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 7, lower bound: -107.3255889, upper bound: 107.3255407
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 7, lower bound: -107.3255797, upper bound: 107.3255297
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 7, lower bound: -107.3258364, upper bound: 107.3257436
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 7, lower bound: -107.3258553, upper bound: 107.3257675
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 7, lower bound: -107.3255459, upper bound: 107.3254933
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 7, lower bound: -107.3255391, upper bound: 107.3254857
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.31
Output dim: 7, lower bound: -107.3277160, upper bound: 107.3284544
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.31
Output dim: 7, lower bound: -107.3281770, upper bound: 107.3281638
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.31
Output dim: 7, lower bound: -107.3284463, upper bound: 107.3284195
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.31
Output dim: 7, lower bound: -107.3281612, upper bound: 107.3281517
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.31
Output dim: 7, lower bound: -107.3276912, upper bound: 107.3284127
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.31
Output dim: 7, lower bound: -107.3280962, upper bound: 107.3280931
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.31
Output dim: 7, lower bound: -107.3283995, upper bound: 107.3283750
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.31
Output dim: 7, lower bound: -107.3280686, upper bound: 107.3280680
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=120.15695190429688
rel_dist={7: [-107.34537841147622, 107.3453783970549]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1810.78 seconds
