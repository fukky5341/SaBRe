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
execution time: IAR + LP analysis = 1.35 + 11.98 = 13.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -107.3456123, upper bound: 107.3456123


# Binary Search by BASE starts (time budget: 1986.67 seconds, max iter: 100)

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
Binary search time: 44.43 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1942.24 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3225538, upper bound: 107.3244613
time: 10.88 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3449864, upper bound: 107.3449864
time: 7.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.81 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 18.81
Output dim: 7, lower bound: -107.3225538, upper bound: 107.3244613
IS_B2, status: Status.UNKNOWN, split count: 1, time: 18.81
Output dim: 7, lower bound: -107.3449864, upper bound: 107.3449864

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -64.8231354, 50.8484344, -55.4400406, 43.5753212, -108.3984528, 106.2884750
1: -53.6626778, 45.5217552, -45.9425201, 38.9366989, -92.5993729, 91.4642563
2: -68.6060944, 42.3251305, -58.1381912, 35.2882538, -103.8943481, 100.4633179
3: -78.9527512, 38.1046982, -68.0392075, 31.9533882, -110.9061356, 106.1439056
4: -70.3417435, 53.6603889, -60.3995552, 46.0324249, -116.3741684, 114.0599136
5: -60.8801041, 47.1619034, -52.0120239, 40.0087280, -100.8888245, 99.1739273
6: -58.2494621, 58.9549713, -49.4500198, 50.8290100, -109.0784607, 108.4049911
7: -67.4825668, 48.6785469, -58.2671661, 40.0144043, -107.4969711, 106.9457092
8: -75.2295532, 50.8719330, -63.5644531, 43.1024208, -118.3319702, 114.4363861
9: -58.1568260, 58.1170082, -49.4447327, 49.5133820, -107.6702118, 107.5617371

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2966489, upper bound: 107.2970078
time: 11.70 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3223623, upper bound: 107.3242970
time: 15.23 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -66.9629440, 52.5338936, -65.1881714, 51.1397667, -118.1027069, 117.7220612
1: -55.4344978, 47.0252304, -53.9649925, 45.7811394, -101.2156296, 100.9902191
2: -70.9811935, 43.8698730, -69.0146408, 42.5899773, -113.5711594, 112.8845139
3: -81.4659729, 39.4653587, -79.3844147, 38.3417397, -119.8077087, 118.8497772
4: -72.6269455, 55.4208984, -70.7281570, 53.9627228, -126.5896530, 126.1490555
5: -62.9200974, 48.7844696, -61.2309990, 47.4403915, -110.3604889, 110.0154724
6: -60.2321777, 60.8447495, -58.5862999, 59.2805290, -119.5127106, 119.4310455
7: -69.6045456, 50.5524025, -67.8424911, 48.9979668, -118.6025085, 118.3948898
8: -77.8515930, 52.6643562, -75.6820450, 51.1857529, -129.0373535, 128.3464050
9: -60.1245384, 60.0685272, -58.4929962, 58.4505005, -118.5750351, 118.5615082

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3273949, upper bound: 107.3261550
time: 12.16 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3448685, upper bound: 107.3448685
time: 8.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.04 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 22.04
Output dim: 7, lower bound: -107.2966489, upper bound: 107.2970078
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 22.04
Output dim: 7, lower bound: -107.3223623, upper bound: 107.3242970
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 22.04
Output dim: 7, lower bound: -107.3273949, upper bound: 107.3261550
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 22.04
Output dim: 7, lower bound: -107.3448685, upper bound: 107.3448685

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -62.1485405, 48.7749939, -54.6720123, 42.9865608, -105.1351013, 103.4469910
1: -51.4544258, 43.5522423, -45.3089027, 38.4051132, -89.8595428, 88.8611450
2: -65.3893585, 39.8909149, -57.3079872, 34.7412071, -100.1305618, 97.1988983
3: -76.0265427, 35.9780617, -67.1281586, 31.4786091, -107.5051346, 103.1062164
4: -67.6138763, 51.4970284, -59.5795708, 45.4130974, -113.0269775, 111.0765991
5: -58.2798615, 44.9777145, -51.2951164, 39.4429169, -97.7227783, 96.2728271
6: -55.5568123, 56.7183571, -48.7433586, 50.1587677, -105.7155762, 105.4617157
7: -64.9664383, 45.5068092, -57.4956779, 39.3579254, -104.3243637, 103.0024872
8: -71.5878220, 48.5055046, -62.6437607, 42.4988098, -114.0866318, 111.1492538
9: -55.5241547, 55.5158272, -48.7462006, 48.8208466, -104.3450012, 104.2620239

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 104

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2883861, upper bound: 107.2882906
time: 11.91 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2895579, upper bound: 107.2900519
time: 16.29 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -63.9659538, 50.1782875, -55.4400406, 43.5753212, -107.5412750, 105.6183243
1: -52.9535637, 44.9243507, -45.9425201, 38.9366989, -91.8902588, 90.8668594
2: -67.6617279, 41.7067566, -58.1381912, 35.2882538, -102.9499664, 99.8449402
3: -77.9456406, 37.5597267, -68.0392075, 31.9533882, -109.8990250, 105.5989380
4: -69.4225998, 52.9594498, -60.3995552, 46.0324249, -115.4550247, 113.3590088
5: -60.0673409, 46.5149078, -52.0120239, 40.0087280, -100.0760651, 98.5269318
6: -57.4553490, 58.1996918, -49.4500198, 50.8290100, -108.2843552, 107.6497116
7: -66.6254654, 47.9289551, -58.2671661, 40.0144043, -106.6398544, 106.1961212
8: -74.1859360, 50.1712112, -63.5644531, 43.1024208, -117.2883606, 113.7356644
9: -57.3711624, 57.3347549, -49.4447327, 49.5133820, -106.8845444, 106.7794800

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3134324, upper bound: 107.3148100
time: 12.56 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3175312, upper bound: 107.3199471
time: 14.38 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -64.0898895, 50.2896042, -64.2427673, 50.4014359, -114.4913254, 114.5323715
1: -53.0554504, 44.9065208, -53.1845856, 45.1229324, -98.1783829, 98.0910950
2: -67.5403671, 41.2985992, -67.9762650, 41.9146461, -109.4550171, 109.2748642
3: -78.3133545, 37.2079010, -78.2720337, 37.7465439, -116.0598831, 115.4799347
4: -69.6880188, 53.0828209, -69.7151031, 53.1905060, -122.8785248, 122.7979279
5: -60.1112175, 46.4402046, -60.3357468, 46.7313766, -106.8425903, 106.7759552
6: -57.3589096, 58.4169083, -57.7146683, 58.4471855, -115.8060913, 116.1315765
7: -66.8935089, 47.2144890, -66.8970413, 48.1831818, -115.0766907, 114.1115189
8: -73.9750671, 50.1169930, -74.5370560, 50.4140625, -124.3891296, 124.6540527
9: -57.3111382, 57.2904510, -57.6286812, 57.5931816, -114.9043198, 114.9191284

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2861017, upper bound: 107.2857015
time: 11.99 seconds

## Relational analysis of IS_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3170266, upper bound: 107.3167190
time: 12.36 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3237308, upper bound: 107.3224107
time: 12.08 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -66.0929947, 51.8533020, -65.1881714, 51.1397667, -117.2327576, 117.0414734
1: -54.7155418, 46.4171753, -53.9649925, 45.7811394, -100.4966812, 100.3821716
2: -70.0227280, 43.2433357, -69.0146408, 42.5899773, -112.6126938, 112.2579727
3: -80.4447098, 38.9130745, -79.3844147, 38.3417397, -118.7864532, 118.2974854
4: -71.6942825, 54.7091408, -70.7281570, 53.9627228, -125.6570053, 125.4373016
5: -62.0950432, 48.1275444, -61.2309990, 47.4403915, -109.5354309, 109.3585358
6: -59.4262390, 60.0777016, -58.5862999, 59.2805290, -118.7067719, 118.6640015
7: -68.7342606, 49.7926636, -67.8424911, 48.9979668, -117.7322235, 117.6351395
8: -76.7948151, 51.9506645, -75.6820450, 51.1857529, -127.9805679, 127.6327057
9: -59.3262596, 59.2746925, -58.4929962, 58.4505005, -117.7767487, 117.7676849

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3192711, upper bound: 107.3202155
time: 13.26 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3103951, upper bound: 107.3103951
time: 6.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.38 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 21.38
Output dim: 7, lower bound: -107.2883861, upper bound: 107.2882906
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 21.38
Output dim: 7, lower bound: -107.2895579, upper bound: 107.2900519
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 21.38
Output dim: 7, lower bound: -107.3134324, upper bound: 107.3148100
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 21.38
Output dim: 7, lower bound: -107.3175312, upper bound: 107.3199471
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 21.38
Output dim: 7, lower bound: -107.3170266, upper bound: 107.3167190
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 21.38
Output dim: 7, lower bound: -107.3237308, upper bound: 107.3224107
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 21.38
Output dim: 7, lower bound: -107.3192711, upper bound: 107.3202155
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 21.38
Output dim: 7, lower bound: -107.3103951, upper bound: 107.3103951

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -50.8246269, 40.0368767, -53.1134872, 41.7924423, -92.6170654, 93.1503525
1: -42.1429291, 35.6527443, -44.0290909, 37.3227730, -79.4656982, 79.6818390
2: -53.0013351, 31.7323074, -55.6148567, 33.6298637, -86.6311951, 87.3471680
3: -62.5989685, 28.8338966, -65.2754288, 30.5049839, -93.1039429, 94.1093140
4: -55.5904961, 42.3371582, -57.9296379, 44.1567039, -99.7472000, 100.2667923
5: -47.6183586, 36.5944901, -49.8296547, 38.2968102, -85.9151688, 86.4241333
6: -45.0726089, 46.8597450, -47.3067741, 48.8024139, -93.8750153, 94.1665192
7: -53.7561188, 35.6581078, -55.9542542, 38.0209007, -91.7770233, 91.6123581
8: -57.8033257, 39.4028397, -60.7582626, 41.2632599, -99.0665741, 100.1611023
9: -45.1842499, 45.3904076, -47.3294678, 47.4377213, -92.6219711, 92.7198715

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2883861, upper bound: 107.2882906
time: 14.63 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2883861, upper bound: 107.2882906
time: 12.49 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -57.9864349, 45.5564041, -54.6720123, 42.9865608, -100.9729919, 100.2283936
1: -48.0419922, 40.6688805, -45.3089027, 38.4051132, -86.4471054, 85.9777832
2: -60.8328476, 36.9267464, -57.3079872, 34.7412071, -95.5740433, 94.2347336
3: -71.1209259, 33.3948631, -67.1281586, 31.4786091, -102.5995331, 100.5230255
4: -63.1867905, 48.1361465, -59.5795708, 45.4130974, -108.5998840, 107.7157135
5: -54.3735657, 41.8999062, -51.2951164, 39.4429169, -93.8164825, 93.1950226
6: -51.7204819, 53.1049004, -48.7433586, 50.1587677, -101.8792419, 101.8482590
7: -60.8430786, 41.9409332, -57.4956779, 39.3579254, -100.2009888, 99.4366150
8: -66.5600662, 45.1478958, -62.6437607, 42.4988098, -109.0588760, 107.7916489
9: -51.7406006, 51.8021164, -48.7462006, 48.8208466, -100.5614471, 100.5483017

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_A1

### Relational analysis result of IS_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2688000, upper bound: 107.2691017
time: 14.65 seconds

## Relational analysis of IS_B1_A1_A2_A2

### Relational analysis result of IS_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2629158, upper bound: 107.2629845
time: 13.99 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -51.1453323, 40.2333832, -53.8646088, 42.3658104, -93.5111313, 94.0979919
1: -42.4232635, 35.9793816, -44.6484909, 37.8415642, -80.2648315, 80.6278687
2: -53.5825615, 32.5109596, -56.4247398, 34.1634178, -87.7459793, 88.9356995
3: -62.8049126, 29.4657707, -66.1669540, 30.9690628, -93.7739716, 95.6327209
4: -55.7957954, 42.5642738, -58.7319641, 44.7627792, -100.5585632, 101.2962341
5: -47.9619331, 37.0080147, -50.5318680, 38.8498688, -86.8117981, 87.5398865
6: -45.6119728, 47.0005112, -47.9997292, 49.4578896, -95.0698624, 95.0002213
7: -53.9416580, 36.8814507, -56.7084351, 38.6632195, -92.6048737, 93.5898743
8: -58.5816345, 39.7698288, -61.6595230, 41.8518944, -100.4335327, 101.4293442
9: -45.7059555, 45.8642654, -48.0141602, 48.1141663, -93.8201065, 93.8784256

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_B1_A2_A1_A1

### Relational analysis result of IS_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3038937, upper bound: 107.3048659
time: 10.92 seconds

## Relational analysis of IS_B1_A2_A1_A2

### Relational analysis result of IS_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3134324, upper bound: 107.3148100
time: 11.85 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -59.3715477, 46.5916138, -55.4400406, 43.5753212, -102.9468689, 102.0316391
1: -49.1810493, 41.7373276, -45.9425201, 38.9366989, -88.1177292, 87.6798401
2: -62.6109543, 38.4369888, -58.1381912, 35.2882538, -97.8991928, 96.5751801
3: -72.5295105, 34.6841469, -68.0392075, 31.9533882, -104.4828873, 102.7233582
4: -64.5222092, 49.2223282, -60.3995552, 46.0324249, -110.5546341, 109.6218796
5: -55.7251244, 43.0996933, -52.0120239, 40.0087280, -95.7338562, 95.1117096
6: -53.2271996, 54.1778336, -49.4500198, 50.8290100, -104.0562134, 103.6278534
7: -62.0745239, 44.0093994, -58.2671661, 40.0144043, -102.0889282, 102.2765656
8: -68.6077423, 46.4109955, -63.5644531, 43.1024208, -111.7101517, 109.9754486
9: -53.2025795, 53.2159081, -49.4447327, 49.5133820, -102.7159576, 102.6606293

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 216

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A2_A2_A1

### Relational analysis result of IS_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3121292, upper bound: 107.3142921
time: 12.95 seconds

## Relational analysis of IS_B1_A2_A2_A2

### Relational analysis result of IS_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3118792, upper bound: 107.3139606
time: 13.31 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -62.2008934, 48.8151169, -51.2936020, 40.3483200, -102.5492096, 100.1087189
1: -51.5055313, 43.5938873, -42.5474243, 36.0852127, -87.5907440, 86.1413040
2: -65.4567413, 39.9467621, -53.7472878, 32.6253700, -98.0821075, 93.6940384
3: -76.0847015, 36.0211983, -62.9808350, 29.5697250, -105.6544266, 99.0020294
4: -67.6830292, 51.5470161, -55.9502869, 42.6871834, -110.3702011, 107.4972992
5: -58.3272705, 45.0378723, -48.1045952, 37.1233101, -95.4505768, 93.1424713
6: -55.6168365, 56.7675056, -45.7521744, 47.1318817, -102.7487183, 102.5196838
7: -65.0269012, 45.5912170, -54.0879669, 37.0233994, -102.0502930, 99.6791840
8: -71.6728363, 48.5603104, -58.7716637, 39.8955154, -111.5683517, 107.3319626
9: -55.5941315, 55.5940437, -45.8468513, 46.0034599, -101.5975723, 101.4408875

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3113701, upper bound: 107.3114188
time: 75.31 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3108647, upper bound: 107.3108758
time: 11.28 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -64.0898895, 50.2896042, -59.6481476, 46.8090782, -110.8989716, 109.9377518
1: -53.0554504, 44.9065208, -49.4090118, 41.9325829, -94.9880371, 94.3155289
2: -67.5403671, 41.2985992, -62.9223557, 38.6443901, -106.1847534, 104.2209549
3: -78.3133545, 37.2079010, -72.8529510, 34.8690643, -113.1824036, 110.0608444
4: -69.6880188, 53.0828209, -64.8136368, 49.4508286, -119.1388474, 117.8964539
5: -60.1112175, 46.4402046, -55.9895287, 43.3118401, -103.4230347, 102.4297333
6: -57.3589096, 58.4169083, -53.4851723, 54.4208412, -111.7797394, 111.9020844
7: -66.8935089, 47.2144890, -62.3446350, 44.2628021, -111.1563110, 109.5591125
8: -73.9750671, 50.1169930, -68.9561996, 46.6500397, -120.6251068, 119.0731812
9: -57.3111382, 57.2904510, -53.4583359, 53.4721222, -110.7832642, 110.7487793

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2802003, upper bound: 107.2798320
time: 14.64 seconds

## Relational analysis of IS_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3189505, upper bound: 107.3166354
time: 13.02 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3189505, upper bound: 107.3224104
time: 12.54 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -65.1904602, 51.1436958, -65.1881714, 51.1397667, -116.3302155, 116.3318558
1: -53.9685326, 45.7865982, -53.9649925, 45.7811394, -99.7496719, 99.7515869
2: -69.0209961, 42.5891953, -69.0146408, 42.5899773, -111.6109695, 111.6038284
3: -79.3834610, 38.3431053, -79.3844147, 38.3417397, -117.7252045, 117.7275085
4: -70.7209625, 53.9664612, -70.7281570, 53.9627228, -124.6836624, 124.6946030
5: -61.2375526, 47.4396782, -61.2309990, 47.4403915, -108.6779480, 108.6706772
6: -58.5831909, 59.2834473, -58.5862999, 59.2805290, -117.8637161, 117.8697510
7: -67.8286438, 48.9954529, -67.8424911, 48.9979668, -116.8266144, 116.8379440
8: -75.6950378, 51.2020264, -75.6820450, 51.1857529, -126.8807907, 126.8840714
9: -58.4900513, 58.4459686, -58.4929962, 58.4505005, -116.9405518, 116.9389648

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_A2_A1_A1

### Relational analysis result of IS_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3121368, upper bound: 107.3128530
time: 14.03 seconds

## Relational analysis of IS_B2_A2_A1_A2

### Relational analysis result of IS_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3192711, upper bound: 107.3202155
time: 12.90 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -61.1254387, 47.9598427, -64.3159561, 50.4539070, -111.5793457, 112.2757797
1: -50.5866241, 42.8599510, -53.2426643, 45.1729774, -95.7595901, 96.1026154
2: -64.1542816, 39.0513496, -68.0465088, 41.9580078, -106.1122894, 107.0978546
3: -74.7949677, 35.3296814, -78.3596802, 37.7901382, -112.5851059, 113.6893539
4: -66.4141541, 50.6074295, -69.7895126, 53.2447281, -119.6588669, 120.3969421
5: -57.2997818, 44.1236687, -60.4021339, 46.7753181, -104.0751038, 104.5258026
6: -54.4699745, 55.8445473, -57.7724457, 58.5136604, -112.9836349, 113.6169891
7: -63.8967590, 44.3860741, -66.9704285, 48.2269669, -112.1237259, 111.3565063
8: -70.3159103, 47.5796280, -74.6162949, 50.4619141, -120.7778168, 122.1959076
9: -54.4694595, 54.4631271, -57.6861115, 57.6499596, -112.1194153, 112.1492157

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 205

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3055465, upper bound: 107.3054945
time: 10.52 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3052845, upper bound: 107.3052845
time: 8.21 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 20.19 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.19
Output dim: 7, lower bound: -107.2883861, upper bound: 107.2882906
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.19
Output dim: 7, lower bound: -107.2883861, upper bound: 107.2882906
IS_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 20.19
Output dim: 7, lower bound: -107.2688000, upper bound: 107.2691017
IS_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 20.19
Output dim: 7, lower bound: -107.2629158, upper bound: 107.2629845
IS_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 20.19
Output dim: 7, lower bound: -107.3038937, upper bound: 107.3048659
IS_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 20.19
Output dim: 7, lower bound: -107.3134324, upper bound: 107.3148100
IS_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 20.19
Output dim: 7, lower bound: -107.3121292, upper bound: 107.3142921
IS_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 20.19
Output dim: 7, lower bound: -107.3118792, upper bound: 107.3139606
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 20.19
Output dim: 7, lower bound: -107.3113701, upper bound: 107.3114188
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 20.19
Output dim: 7, lower bound: -107.3108647, upper bound: 107.3108758
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 20.19
Output dim: 7, lower bound: -107.3189505, upper bound: 107.3166354
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 20.19
Output dim: 7, lower bound: -107.3189505, upper bound: 107.3224104
IS_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 20.19
Output dim: 7, lower bound: -107.3121368, upper bound: 107.3128530
IS_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 20.19
Output dim: 7, lower bound: -107.3192711, upper bound: 107.3202155
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.19
Output dim: 7, lower bound: -107.3055465, upper bound: 107.3054945
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.19
Output dim: 7, lower bound: -107.3052845, upper bound: 107.3052845

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -50.8246269, 40.0368767, -45.2637024, 35.7168198, -86.5414429, 85.3005676
1: -42.1429291, 35.6527443, -37.5153847, 31.7983036, -73.9412308, 73.1681290
2: -53.0013351, 31.7323074, -47.0304108, 27.9637527, -80.9650879, 78.7627182
3: -62.5989685, 28.8338966, -55.8355942, 25.5222912, -88.1212616, 84.6694870
4: -55.5904961, 42.3371582, -49.5548706, 37.7556458, -93.3461456, 91.8920212
5: -47.6183586, 36.5944901, -42.3908806, 32.5044022, -80.1227570, 78.9853668
6: -45.0726089, 46.8597450, -39.9563370, 41.9330635, -87.0056610, 86.8160858
7: -53.7561188, 35.6581078, -48.1042061, 31.1078281, -84.8639450, 83.7623062
8: -57.8033257, 39.4028397, -51.1792526, 34.9697647, -92.7730789, 90.5820923
9: -45.1842499, 45.3904076, -40.1207123, 40.3515625, -85.5358124, 85.5111237

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2821168, upper bound: 107.2821923
time: 11.68 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2816103, upper bound: 107.2816633
time: 14.43 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -50.8246269, 40.0368767, -50.9860039, 40.1564140, -90.9810410, 91.0228806
1: -42.1429291, 35.6527443, -42.2735901, 35.8506889, -77.9936218, 77.9263306
2: -53.0013351, 31.7323074, -53.3056717, 32.1153145, -85.1166534, 85.0379791
3: -62.5989685, 28.8338966, -62.7363968, 29.1915054, -91.7904739, 91.5702896
4: -55.5904961, 42.3371582, -55.6669464, 42.4370346, -98.0275269, 98.0040741
5: -47.6183586, 36.5944901, -47.8306770, 36.7353172, -84.3536758, 84.4251556
6: -45.0726089, 46.8597450, -45.3435631, 46.9499283, -92.0225296, 92.2033081
7: -53.7561188, 35.6581078, -53.8339767, 36.1917801, -89.9478989, 89.4920807
8: -57.8033257, 39.4028397, -58.1885529, 39.5850220, -97.3883362, 97.5913925
9: -45.1842499, 45.3904076, -45.3857498, 45.5346489, -90.7188950, 90.7761459

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2827388, upper bound: 107.2827066
time: 15.26 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2816103, upper bound: 107.2816633
time: 13.99 seconds

## BFS IS instance: IS_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -53.4256973, 42.0623322, -54.6720123, 42.9865608, -96.4122543, 96.7343292
1: -44.2840462, 37.5267792, -45.3089027, 38.4051132, -82.6891556, 82.8356781
2: -55.9003296, 33.6865768, -57.3079872, 34.7412071, -90.6415405, 90.9945602
3: -65.7178574, 30.6372318, -67.1281586, 31.4786091, -97.1964417, 97.7653885
4: -58.3054543, 44.4544296, -59.5795708, 45.4130974, -103.7185516, 104.0339966
5: -50.1043015, 38.5251122, -51.2951164, 39.4429169, -89.5472183, 89.8202286
6: -47.5010986, 49.1548691, -48.7433586, 50.1587677, -97.6598587, 97.8982239
7: -56.3193817, 38.0224609, -57.4956779, 39.3579254, -95.6773071, 95.5181351
8: -61.0748749, 41.5466652, -62.6437607, 42.4988098, -103.5736847, 104.1904221
9: -47.5869217, 47.7525101, -48.7462006, 48.8208466, -96.4077682, 96.4986954

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_A2_A1_B1

### Relational analysis result of IS_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2688003, upper bound: 107.2691017
time: 8.56 seconds

## Relational analysis of IS_B1_A1_A2_A1_B2

### Relational analysis result of IS_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2688003, upper bound: 107.2691017
time: 12.81 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.79 seconds
IS_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.79
Output dim: 7, lower bound: -107.2821168, upper bound: 107.2821923
IS_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.79
Output dim: 7, lower bound: -107.2816103, upper bound: 107.2816633
IS_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 22.79
Output dim: 7, lower bound: -107.2827388, upper bound: 107.2827066
IS_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 22.79
Output dim: 7, lower bound: -107.2816103, upper bound: 107.2816633
IS_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.79
Output dim: 7, lower bound: -107.2688003, upper bound: 107.2691017
IS_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.79
Output dim: 7, lower bound: -107.2688003, upper bound: 107.2691017
IS_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -107.2629158, upper bound: 107.2629845
IS_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -107.3038937, upper bound: 107.3048659
IS_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -107.3134324, upper bound: 107.3148100
IS_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -107.3121292, upper bound: 107.3142921
IS_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -107.3118792, upper bound: 107.3139606
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -107.3113701, upper bound: 107.3114188
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -107.3108647, upper bound: 107.3108758
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -107.3189505, upper bound: 107.3166354
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -107.3189505, upper bound: 107.3224104
IS_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -107.3121368, upper bound: 107.3128530
IS_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -107.3192711, upper bound: 107.3202155
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -107.3055465, upper bound: 107.3054945
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.79
Output dim: 7, lower bound: -107.3052845, upper bound: 107.3052845
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=120.15695190429688
rel_dist={7: [-107.34558116528925, 107.34558116528925]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3241953, upper bound: 107.3248325
time: 13.95 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3453684, upper bound: 107.3453684
time: 11.22 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 25.31 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 25.31
Output dim: 7, lower bound: -107.3241953, upper bound: 107.3248325
IS_B2, status: Status.UNKNOWN, split count: 1, time: 25.31
Output dim: 7, lower bound: -107.3453684, upper bound: 107.3453684

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -64.6237183, 50.7071915, -64.0898895, 50.2896042, -114.9133224, 114.7970734
1: -53.5045700, 45.3951416, -53.0554504, 44.9065208, -98.4110870, 98.4505920
2: -68.4125214, 42.2013855, -67.5403671, 41.2985992, -109.7111206, 109.7417526
3: -78.7144165, 37.9930382, -78.3133545, 37.2079010, -115.9223175, 116.3063965
4: -70.1204071, 53.5102806, -69.6880188, 53.0828209, -123.2032318, 123.1983032
5: -60.7049866, 47.0303459, -60.1112175, 46.4402046, -107.1451874, 107.1415558
6: -58.0755424, 58.7820549, -57.3589096, 58.4169083, -116.4924316, 116.1409607
7: -67.2641296, 48.5393410, -66.8935089, 47.2144890, -114.4786224, 115.4328461
8: -75.0211639, 50.7537689, -73.9750671, 50.1169930, -125.1381378, 124.7288361
9: -57.9863510, 57.9476929, -57.3111382, 57.2904510, -115.2768021, 115.2588348

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 216

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2740920, upper bound: 107.2737537
time: 13.70 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3234419, upper bound: 107.3240246
time: 11.15 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -66.9629440, 52.5338936, -66.0929947, 51.8533020, -118.8162460, 118.6268921
1: -55.4344978, 47.0252304, -54.7155418, 46.4171753, -101.8516541, 101.7407532
2: -70.9811935, 43.8698730, -70.0227280, 43.2433357, -114.2245255, 113.8926010
3: -81.4659729, 39.4653587, -80.4447098, 38.9130745, -120.3790436, 119.9100647
4: -72.6269455, 55.4208984, -71.6942825, 54.7091408, -127.3360901, 127.1151810
5: -62.9200974, 48.7844696, -62.0950432, 48.1275444, -111.0476379, 110.8795166
6: -60.2321777, 60.8447495, -59.4262390, 60.0777016, -120.3098755, 120.2709885
7: -69.6045456, 50.5524025, -68.7342606, 49.7926636, -119.3972015, 119.2866669
8: -77.8515930, 52.6643562, -76.7948151, 51.9506645, -129.8022461, 129.4591675
9: -60.1245384, 60.0685272, -59.3262596, 59.2746925, -119.3992310, 119.3947601

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3200546, upper bound: 107.3187724
time: 11.60 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3447910, upper bound: 107.3447911
time: 11.08 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.10 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 24.10
Output dim: 7, lower bound: -107.2740920, upper bound: 107.2737537
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 24.10
Output dim: 7, lower bound: -107.3234419, upper bound: 107.3240246
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 24.10
Output dim: 7, lower bound: -107.3200546, upper bound: 107.3187724
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 24.10
Output dim: 7, lower bound: -107.3447910, upper bound: 107.3447911

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -53.5391731, 42.1203423, -59.5507774, 46.7659111, -100.3050842, 101.6711197
1: -44.3740349, 37.6215553, -49.3179817, 41.7450447, -86.1190796, 86.9395370
2: -56.0859833, 33.9338531, -62.5318451, 38.0151443, -94.1011276, 96.4656982
3: -65.7847443, 30.7790642, -72.9798584, 34.3405380, -100.1252747, 103.7589188
4: -58.3692551, 44.4997597, -64.8476639, 49.3855057, -107.7547607, 109.3474121
5: -50.2370682, 38.6084862, -55.8361015, 43.0304718, -93.2675400, 94.4445877
6: -47.6998291, 49.1702347, -53.1514702, 54.4616661, -102.1614990, 102.3217010
7: -56.3584328, 38.3885193, -62.3917923, 43.2292976, -99.5877228, 100.7803116
8: -61.2847900, 41.6106949, -68.4241180, 46.3814850, -107.6662750, 110.0348129
9: -47.7137146, 47.8003654, -53.1436043, 53.1605835, -100.8742981, 100.9439697

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2579369, upper bound: 107.2576650
time: 13.54 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546676
time: 12.90 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -62.8776627, 49.3370132, -64.0898895, 50.2896042, -113.1672668, 113.4269028
1: -52.0572395, 44.1745987, -53.0554504, 44.9065208, -96.9637604, 97.2300491
2: -66.4781494, 40.9402580, -67.5403671, 41.2985992, -107.7767487, 108.4806213
3: -76.6658783, 36.8865433, -78.3133545, 37.2079010, -113.8737793, 115.1998901
4: -68.2523270, 52.0759048, -69.6880188, 53.0828209, -121.3351440, 121.7639160
5: -59.0433197, 45.7090378, -60.1112175, 46.4402046, -105.4835205, 105.8202438
6: -56.4564171, 57.2446365, -57.3589096, 58.4169083, -114.8733215, 114.6035385
7: -65.5315933, 47.0089111, -66.8935089, 47.2144890, -112.7460785, 113.9024200
8: -72.8826294, 49.3036499, -73.9750671, 50.1169930, -122.9996033, 123.2787018
9: -56.3835754, 56.3552704, -57.3111382, 57.2904510, -113.6740265, 113.6663895

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 216

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3183282, upper bound: 107.3187605
time: 13.05 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
time: 15.08 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -55.4400406, 43.5753212, -61.0351028, 47.8798599, -103.3198929, 104.6104279
1: -45.9425201, 38.9366989, -50.5320473, 42.8757095, -88.8182144, 89.4687347
2: -58.1381912, 35.2882538, -64.4145279, 39.5853996, -97.7235870, 99.7027740
3: -68.0392075, 31.9533882, -74.5025253, 35.6972809, -103.7364883, 106.4559097
4: -60.3995552, 46.0324249, -66.2943497, 50.5535736, -110.9531174, 112.3267746
5: -52.0120239, 40.0087280, -57.2852669, 44.2979431, -96.3099670, 97.2939911
6: -49.4500198, 50.8290100, -54.7430878, 55.6221161, -105.0721359, 105.5720978
7: -58.2671661, 40.0144043, -63.7233238, 45.3617668, -103.6289368, 103.7377319
8: -63.5644531, 43.1024208, -70.5862198, 47.7337456, -111.2982025, 113.6886444
9: -49.4447327, 49.5133820, -54.6842041, 54.6602631, -104.1049805, 104.1975861

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 86

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3140047, upper bound: 107.3130453
time: 13.43 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3138109, upper bound: 107.3129120
time: 15.18 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -65.1881714, 51.1397667, -66.0929947, 51.8533020, -117.0414734, 117.2327576
1: -53.9649925, 45.7811394, -54.7155418, 46.4171753, -100.3821716, 100.4966812
2: -69.0146408, 42.5899773, -70.0227280, 43.2433357, -112.2579803, 112.6126938
3: -79.3844147, 38.3417397, -80.4447098, 38.9130745, -118.2974854, 118.7864532
4: -70.7281570, 53.9627228, -71.6942825, 54.7091408, -125.4373016, 125.6570053
5: -61.2309990, 47.4403915, -62.0950432, 48.1275444, -109.3585358, 109.5354309
6: -58.5862999, 59.2805290, -59.4262390, 60.0777016, -118.6640015, 118.7067719
7: -67.8424911, 48.9979668, -68.7342606, 49.7926636, -117.6351471, 117.7322235
8: -75.6820450, 51.1857529, -76.7948151, 51.9506645, -127.6327057, 127.9805679
9: -58.4929962, 58.4505005, -59.3262596, 59.2746925, -117.7676849, 117.7767487

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3187724, upper bound: 107.3200545
time: 10.99 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3187724, upper bound: 107.3447910
time: 14.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 27.01 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 27.01
Output dim: 7, lower bound: -107.2579369, upper bound: 107.2576650
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 27.01
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546676
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 27.01
Output dim: 7, lower bound: -107.3183282, upper bound: 107.3187605
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 27.01
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 27.01
Output dim: 7, lower bound: -107.3140047, upper bound: 107.3130453
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 27.01
Output dim: 7, lower bound: -107.3138109, upper bound: 107.3129120
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 27.01
Output dim: 7, lower bound: -107.3187724, upper bound: 107.3200545
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 27.01
Output dim: 7, lower bound: -107.3187724, upper bound: 107.3447910

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -50.0276489, 39.4127998, -49.3819962, 38.9139671, -88.9416199, 88.7947998
1: -41.4683533, 35.1826515, -40.9076614, 34.6798820, -76.1482315, 76.0903168
2: -52.2912865, 31.4628181, -51.5208740, 30.8556442, -83.1469269, 82.9836807
3: -61.5907745, 28.6297302, -60.8525848, 28.1191196, -89.7098846, 89.4823151
4: -54.6312523, 41.6408997, -53.9873657, 41.1333084, -95.7645569, 95.6282654
5: -46.9321518, 36.0331650, -46.2984886, 35.5409775, -82.4731216, 82.3316498
6: -44.4621964, 46.0926743, -43.7853661, 45.5543671, -90.0165405, 89.8780365
7: -52.8490562, 35.4106216, -52.2397881, 34.6011238, -87.4501801, 87.6504059
8: -57.0591049, 38.8224945, -56.1661568, 38.2845116, -95.3436127, 94.9886398
9: -44.5140648, 44.6539192, -43.8883514, 44.0585060, -88.5725708, 88.5422668

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546676
time: 11.99 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546675
time: 13.83 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -49.6825256, 39.1432190, -53.4594879, 42.0734634, -91.7559814, 92.6026840
1: -41.1767044, 34.9473190, -44.2839775, 37.5119019, -78.6886063, 79.2312927
2: -51.9238892, 31.2349396, -55.8829880, 33.5841904, -85.5080795, 87.1179276
3: -61.1632538, 28.4309635, -65.7723694, 30.5239182, -91.6871490, 94.2033310
4: -54.2514420, 41.3556595, -58.3834763, 44.4662552, -98.7176971, 99.7391357
5: -46.6098480, 35.7804070, -50.1366920, 38.4875183, -85.0973663, 85.9170990
6: -44.1459885, 45.7858353, -47.4961777, 49.1758461, -93.3218384, 93.2820129
7: -52.4942513, 35.1438255, -56.3817825, 37.8290520, -90.3233032, 91.5256042
8: -56.6528397, 38.5518036, -60.9906616, 41.4725952, -98.1254349, 99.5424576
9: -44.2027626, 44.3368607, -47.5617218, 47.6706161, -91.8733826, 91.8985748

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546676
time: 13.35 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546676
time: 15.52 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -55.0805397, 43.2661705, -61.7262878, 48.4415283, -103.5220642, 104.9924622
1: -45.6307144, 38.7455635, -51.1072388, 43.2606697, -88.8913879, 89.8527985
2: -57.8556366, 35.2947693, -64.9229736, 39.5841904, -97.4398270, 100.2177429
3: -67.4692230, 32.0212402, -75.5256958, 35.7320709, -103.2012939, 107.5469360
4: -59.8202209, 45.7318497, -67.1389236, 51.1586914, -110.9789124, 112.8707504
5: -51.7342415, 39.8341408, -57.8923607, 44.6524887, -96.3867188, 97.7265015
6: -49.1844482, 50.4492493, -55.1557236, 56.3599472, -105.5443878, 105.6049728
7: -57.7582130, 40.1201515, -64.5409241, 45.1181374, -102.8763351, 104.6610718
8: -63.3778915, 42.9164085, -71.0867386, 48.1603813, -111.5382690, 114.0031433
9: -49.1741829, 49.1985130, -55.1209984, 55.1156845, -104.2898560, 104.3195114

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 216

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
time: 11.19 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
time: 11.39 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -57.1877594, 44.9243965, -60.0677338, 47.1601372, -104.3479004, 104.9921265
1: -47.3940163, 40.1773949, -49.7498970, 42.1107483, -89.5047607, 89.9272919
2: -60.0249596, 36.5409088, -63.0942688, 38.3915825, -98.4165344, 99.6351700
3: -70.0608826, 33.1320763, -73.5752335, 34.6973419, -104.7582169, 106.7073059
4: -62.1451302, 47.4573021, -65.3589172, 49.8145943, -111.9597244, 112.8162079
5: -53.6910744, 41.3198776, -56.3403244, 43.4122086, -97.1032715, 97.6602020
6: -51.0217438, 52.3903999, -53.6124458, 54.9290619, -105.9508057, 106.0028458
7: -59.9924812, 41.4736404, -62.9004860, 43.6638412, -103.6563263, 104.3741302
8: -65.7246094, 44.5008316, -69.0672684, 46.8089638, -112.5335693, 113.5681000
9: -51.0037651, 51.0681381, -53.5991135, 53.6161346, -104.6199036, 104.6672516

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
time: 12.65 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
time: 13.34 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -53.4594040, 42.0575905, -53.6024170, 42.1242943, -95.5836792, 95.6599884
1: -44.3083839, 37.5598564, -44.4081001, 37.7041664, -82.0125504, 81.9679489
2: -55.9695435, 33.8524208, -56.2152405, 34.2056999, -90.1752472, 90.0676575
3: -65.6860657, 30.7276421, -65.7274780, 31.0680180, -96.7540817, 96.4551239
4: -58.2664261, 44.4310760, -58.2560844, 44.5273972, -102.7938232, 102.6871643
5: -50.1671524, 38.5275841, -50.3388367, 38.7138901, -88.8810349, 88.8664093
6: -47.6005859, 49.1146393, -47.8055000, 49.1658745, -96.7664642, 96.9201355
7: -56.2915726, 38.2549744, -56.3047485, 38.7850494, -95.0766144, 94.5597153
8: -61.1633759, 41.5232773, -61.5460434, 41.7067795, -102.8701553, 103.0693207
9: -47.6115761, 47.7037697, -47.8107719, 47.8497505, -95.4613113, 95.5145416

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3138109, upper bound: 107.3129119
time: 10.64 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3138109, upper bound: 107.3129119
time: 12.73 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -52.0717468, 40.9922142, -55.7002945, 43.7746315, -95.8463745, 96.6925049
1: -43.1643944, 36.5972443, -46.1646004, 39.1295052, -82.2938995, 82.7618408
2: -54.4513206, 32.8469048, -58.3737106, 35.4433861, -89.8947067, 91.2205811
3: -64.0302887, 29.8596706, -68.3081207, 32.1714134, -96.2017059, 98.1677856
4: -56.7753792, 43.3057709, -60.5714493, 46.2454567, -103.0208359, 103.8772125
5: -48.8648605, 37.4889679, -52.2893524, 40.1897888, -89.0546417, 89.7783051
6: -46.3000412, 47.9144058, -49.6324806, 51.0984268, -97.3984680, 97.5468903
7: -54.9171181, 37.0230522, -58.5302277, 40.1267052, -95.0438232, 95.5532837
8: -59.4663620, 40.4158440, -63.8801079, 43.2821007, -102.7484589, 104.2959366
9: -46.3323021, 46.4471779, -49.6284027, 49.7106934, -96.0429764, 96.0755768

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3081613, upper bound: 107.3081613
time: 10.34 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3081613, upper bound: 107.3129120
time: 10.67 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -65.1881714, 51.1397667, -54.7550430, 43.0498543, -108.2380219, 105.8948059
1: -53.9649925, 45.7811394, -45.3764496, 38.4613228, -92.4263153, 91.1575928
2: -69.0146408, 42.5899773, -57.3954163, 34.7940369, -103.8086777, 99.9853973
3: -79.3844147, 38.3417397, -67.2282028, 31.5259132, -110.9103088, 105.5699463
4: -70.7281570, 53.9627228, -59.6685715, 45.4804764, -116.2086334, 113.6312866
5: -61.2309990, 47.4403915, -51.3724365, 39.5012093, -100.7322083, 98.8128281
6: -58.5862999, 59.2805290, -48.8163567, 50.2317276, -108.8180237, 108.0968857
7: -67.8424911, 48.9979668, -57.5789642, 39.4182701, -107.2607574, 106.5769348
8: -75.6820450, 51.1857529, -62.7391891, 42.5629730, -118.2450104, 113.9249420
9: -58.4929962, 58.4505005, -48.8188286, 48.8914185, -107.3843994, 107.2693253

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3130453, upper bound: 107.3140047
time: 14.76 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3129120, upper bound: 107.3138109
time: 12.31 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -65.1881714, 51.1397667, -64.3235779, 50.4634399, -115.6516113, 115.4633484
1: -53.9649925, 45.7811394, -53.2500916, 45.1778107, -99.1428070, 99.0312347
2: -69.0146408, 42.5899773, -68.0623703, 41.9664879, -110.9811249, 110.6523438
3: -79.3844147, 38.3417397, -78.3687363, 37.7928085, -117.1772232, 116.7104797
4: -70.7281570, 53.9627228, -69.8010712, 53.2558861, -123.9840393, 123.7637863
5: -61.2309990, 47.4403915, -60.4110146, 46.7876472, -108.0186462, 107.8514099
6: -58.5862999, 59.2805290, -57.7851524, 58.5186501, -117.1049500, 117.0656815
7: -67.8424911, 48.9979668, -66.9783173, 48.2418861, -116.0843811, 115.9762878
8: -75.6820450, 51.1857529, -74.6313095, 50.4773140, -126.1593628, 125.8170624
9: -58.4929962, 58.4505005, -57.6994781, 57.6620560, -116.1550522, 116.1499786

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3130453, upper bound: 107.3412111
time: 11.46 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3129120, upper bound: 107.3410193
time: 13.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.55 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 26.55
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546676
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 26.55
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546675
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 26.55
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546676
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 26.55
Output dim: 7, lower bound: -107.2550427, upper bound: 107.2546676
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.55
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.55
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.55
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.55
Output dim: 7, lower bound: -107.3181498, upper bound: 107.3185540
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 26.55
Output dim: 7, lower bound: -107.3138109, upper bound: 107.3129119
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 26.55
Output dim: 7, lower bound: -107.3138109, upper bound: 107.3129119
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 26.55
Output dim: 7, lower bound: -107.3081613, upper bound: 107.3081613
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 26.55
Output dim: 7, lower bound: -107.3081613, upper bound: 107.3129120
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 26.55
Output dim: 7, lower bound: -107.3130453, upper bound: 107.3140047
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 26.55
Output dim: 7, lower bound: -107.3129120, upper bound: 107.3138109
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 26.55
Output dim: 7, lower bound: -107.3130453, upper bound: 107.3412111
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 26.55
Output dim: 7, lower bound: -107.3129120, upper bound: 107.3410193

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -44.3392677, 34.9923286, -49.3819962, 38.9139671, -83.2532349, 84.3743286
1: -36.7083855, 31.1916733, -40.9076614, 34.6798820, -71.3882523, 72.0993347
2: -46.1251526, 27.4609241, -51.5208740, 30.8556442, -76.9807968, 78.9817886
3: -54.7107964, 25.1574898, -60.8525848, 28.1191196, -82.8299026, 86.0100708
4: -48.5052795, 36.9826813, -53.9873657, 41.1333084, -89.6385880, 90.9700470
5: -41.5625000, 31.8541451, -46.2984886, 35.5409775, -77.1034698, 78.1526337
6: -39.1809349, 41.0779877, -43.7853661, 45.5543671, -84.7352905, 84.8633499
7: -47.1004639, 30.5437088, -52.2397881, 34.6011238, -81.7015686, 82.7834930
8: -50.1820297, 34.2791519, -56.1661568, 38.2845116, -88.4665375, 90.4452972
9: -39.3186569, 39.5061722, -43.8883514, 44.0585060, -83.3771591, 83.3945084

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2508838, upper bound: 107.2505133
time: 11.88 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.2512934, upper bound: 107.2509116
time: 16.24 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -48.2880096, 38.0674820, -49.3819962, 38.9139671, -87.2019806, 87.4494781
1: -40.0008392, 33.9532394, -40.9076614, 34.6798820, -74.6807175, 74.8609009
2: -50.3583488, 30.1097832, -51.5208740, 30.8556442, -81.2139893, 81.6306534
3: -59.5093460, 27.4788666, -60.8525848, 28.1191196, -87.6284637, 88.3314514
4: -52.7964630, 40.2217064, -53.9873657, 41.1333084, -93.9297714, 94.2090759
5: -45.2866135, 34.7168121, -46.2984886, 35.5409775, -80.8275909, 81.0152969
6: -42.7999115, 44.5997887, -43.7853661, 45.5543671, -88.3542709, 88.3851471
7: -51.1472740, 33.6924286, -52.2397881, 34.6011238, -85.7483978, 85.9322205
8: -54.8588371, 37.3780212, -56.1661568, 38.2845116, -93.1433487, 93.5441742
9: -42.8854942, 43.0283356, -43.8883514, 44.0585060, -86.9440002, 86.9166870

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=120.15695190429688
rel_dist={7: [-107.34546615888509, 107.34546613728932]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3418282, upper bound: 107.3418112
time: 10.55 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3417525, upper bound: 107.3417522
time: 9.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 20.29 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 20.29
Output dim: 7, lower bound: -107.3418282, upper bound: 107.3418112
IS_B2, status: Status.UNKNOWN, split count: 1, time: 20.29
Output dim: 7, lower bound: -107.3417525, upper bound: 107.3417522

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -60.8302460, 47.7042122, -58.6679535, 46.0280190, -106.8582611, 106.3721542
1: -50.3648796, 42.7354012, -48.5801163, 41.2324944, -91.5973663, 91.3155136
2: -64.1760254, 39.4295654, -61.7794418, 37.8637924, -102.0398026, 101.2090073
3: -74.2580872, 35.6145363, -71.6995926, 34.2636337, -108.5217209, 107.3141174
4: -66.0081024, 50.3909149, -63.6653938, 48.6293106, -114.6374130, 114.0562973
5: -57.1162796, 44.1227722, -55.0948219, 42.4995308, -99.6157913, 99.2175903
6: -54.5172005, 55.4667969, -52.4983368, 53.5801926, -108.0973816, 107.9651337
7: -63.5089340, 45.1283340, -61.3495483, 43.2161179, -106.7250519, 106.4778824
8: -70.3558578, 47.5615921, -67.7140045, 45.7863121, -116.1421661, 115.2755890
9: -54.4528732, 54.4303246, -52.4486389, 52.4439316, -106.8968048, 106.8789368

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 119

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3417525, upper bound: 107.3417522
time: 11.42 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3417525, upper bound: 107.3417525
time: 10.27 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -59.4065437, 46.6050224, -60.7526093, 47.6667824, -107.0733261, 107.3576202
1: -49.2009239, 41.7495270, -50.3260536, 42.6485825, -91.8495026, 92.0755768
2: -62.6001778, 38.4135971, -63.9215698, 39.0961494, -101.6963196, 102.3351669
3: -72.5768280, 34.7157211, -74.2678375, 35.3619385, -107.9387665, 108.9835587
4: -64.4872742, 49.2356415, -65.9651184, 50.3381081, -114.8253784, 115.2007599
5: -55.7793617, 43.0641403, -57.0271950, 43.9688683, -99.7482300, 100.0913391
6: -53.1991386, 54.2309532, -54.3141594, 55.4990654, -108.6982040, 108.5451126
7: -62.1130447, 43.8955994, -63.5629807, 44.5565109, -106.6695404, 107.4585800
8: -68.6097336, 46.3918457, -70.0351562, 47.3521004, -115.9618225, 116.4269943
9: -53.1576157, 53.1566200, -54.2605591, 54.2926788, -107.4502945, 107.4171753

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 232

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3417525, upper bound: 107.3417525
time: 9.84 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3417525, upper bound: 107.3417522
time: 9.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 20.96 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 20.96
Output dim: 7, lower bound: -107.3417525, upper bound: 107.3417522
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 20.96
Output dim: 7, lower bound: -107.3417525, upper bound: 107.3417525
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 20.96
Output dim: 7, lower bound: -107.3417525, upper bound: 107.3417525
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 20.96
Output dim: 7, lower bound: -107.3417525, upper bound: 107.3417522

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -58.6679535, 46.0280190, -58.6679535, 46.0280190, -104.6959686, 104.6959686
1: -48.5801163, 41.2324944, -48.5801163, 41.2324944, -89.8126068, 89.8126068
2: -61.7794418, 37.8637924, -61.7794418, 37.8637924, -99.6432343, 99.6432343
3: -71.6995926, 34.2636337, -71.6995926, 34.2636337, -105.9632263, 105.9632263
4: -63.6653938, 48.6293106, -63.6653938, 48.6293106, -112.2947083, 112.2947083
5: -55.0948219, 42.4995308, -55.0948219, 42.4995308, -97.5943451, 97.5943451
6: -52.4983368, 53.5801926, -52.4983368, 53.5801926, -106.0785294, 106.0785294
7: -61.3495483, 43.2161179, -61.3495483, 43.2161179, -104.5656662, 104.5656662
8: -67.7140045, 45.7863121, -67.7140045, 45.7863121, -113.5003204, 113.5003128
9: -52.4486389, 52.4439316, -52.4486389, 52.4439316, -104.8925552, 104.8925552

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3328072, upper bound: 107.3327111
time: 9.68 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3324445, upper bound: 107.3324206
time: 10.04 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -60.7526093, 47.6667824, -58.6679535, 46.0280190, -106.7806244, 106.3347321
1: -50.3260536, 42.6485825, -48.5801163, 41.2324944, -91.5585480, 91.2286835
2: -63.9215698, 39.0961494, -61.7794418, 37.8637924, -101.7853546, 100.8755951
3: -74.2678375, 35.3619385, -71.6995926, 34.2636337, -108.5314713, 107.0615311
4: -65.9651184, 50.3381081, -63.6653938, 48.6293106, -114.5944290, 114.0035019
5: -57.0271950, 43.9688683, -55.0948219, 42.4995308, -99.5267181, 99.0636902
6: -54.3141594, 55.4990654, -52.4983368, 53.5801926, -107.8943481, 107.9974060
7: -63.5629807, 44.5565109, -61.3495483, 43.2161179, -106.7790985, 105.9060593
8: -70.0351562, 47.3521004, -67.7140045, 45.7863121, -115.8214722, 115.0661011
9: -54.2605591, 54.2926788, -52.4486389, 52.4439316, -106.7044907, 106.7412872

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 232

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3327572, upper bound: 107.3328147
time: 11.73 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3324445, upper bound: 107.3324206
time: 10.45 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -58.6600342, 46.0218887, -60.7526093, 47.6667824, -106.3267975, 106.7744827
1: -48.5733070, 41.2270088, -50.3260536, 42.6485825, -91.2218781, 91.5530624
2: -61.7705307, 37.8582039, -63.9215698, 39.0961494, -100.8666840, 101.7797623
3: -71.6901093, 34.2585487, -74.2678375, 35.3619385, -107.0520325, 108.5263824
4: -63.6565704, 48.6228409, -65.9651184, 50.3381081, -113.9946747, 114.5879517
5: -55.0872879, 42.4935760, -57.0271950, 43.9688683, -99.0561447, 99.5207672
6: -52.4908180, 53.5733528, -54.3141594, 55.4990654, -107.9898834, 107.8875122
7: -61.3414688, 43.2089767, -63.5629807, 44.5565109, -105.8979797, 106.7719574
8: -67.7038574, 45.7799835, -70.0351562, 47.3521004, -115.0559387, 115.8151398
9: -52.4411469, 52.4367218, -54.2605591, 54.2926788, -106.7338181, 106.6972809

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 232

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3327321, upper bound: 107.3326598
time: 13.89 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3323532, upper bound: 107.3323532
time: 11.56 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -60.7524490, 47.6666794, -60.7526093, 47.6667824, -108.4192123, 108.4192810
1: -50.3259354, 42.6484871, -50.3260536, 42.6485825, -92.9745102, 92.9745331
2: -63.9214058, 39.0960541, -63.9215698, 39.0961494, -103.0175552, 103.0176163
3: -74.2676239, 35.3618736, -74.2678375, 35.3619385, -109.6295547, 109.6296997
4: -65.9650116, 50.3379822, -65.9651184, 50.3381081, -116.3031158, 116.3031006
5: -57.0270615, 43.9687843, -57.0271950, 43.9688683, -100.9959259, 100.9959717
6: -54.3140030, 55.4989662, -54.3141594, 55.4990654, -109.8130646, 109.8131256
7: -63.5628586, 44.5563698, -63.5629807, 44.5565109, -108.1193695, 108.1193542
8: -70.0349808, 47.3520050, -70.0351562, 47.3521004, -117.3870621, 117.3871613
9: -54.2604485, 54.2925415, -54.2605591, 54.2926788, -108.5531235, 108.5531006

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3327321, upper bound: 107.3326598
time: 12.62 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3323532, upper bound: 107.3323532
time: 10.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.16 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 24.16
Output dim: 7, lower bound: -107.3328072, upper bound: 107.3327111
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 24.16
Output dim: 7, lower bound: -107.3324445, upper bound: 107.3324206
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 24.16
Output dim: 7, lower bound: -107.3327572, upper bound: 107.3328147
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 24.16
Output dim: 7, lower bound: -107.3324445, upper bound: 107.3324206
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 24.16
Output dim: 7, lower bound: -107.3327321, upper bound: 107.3326598
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 24.16
Output dim: 7, lower bound: -107.3323532, upper bound: 107.3323532
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 24.16
Output dim: 7, lower bound: -107.3327321, upper bound: 107.3326598
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 24.16
Output dim: 7, lower bound: -107.3323532, upper bound: 107.3323532

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -55.4533615, 43.5631256, -53.7935791, 42.2900391, -97.7433853, 97.3567047
1: -45.9522514, 39.0272293, -44.5852928, 37.8822327, -83.8344879, 83.6125183
2: -58.2908936, 35.6041222, -56.4886551, 34.4299355, -92.7208252, 92.0927582
3: -67.9119186, 32.3178139, -65.9439926, 31.3121815, -99.2240982, 98.2617950
4: -60.2234650, 46.0333633, -58.4458122, 44.6872253, -104.9106903, 104.4791718
5: -52.0753822, 40.1334343, -50.5194397, 38.9095802, -90.9849472, 90.6528778
6: -49.5322342, 50.7984772, -47.9954147, 49.3605423, -98.8927765, 98.7938919
7: -58.1627274, 40.4947624, -56.5085754, 39.0789452, -97.2416534, 97.0033417
8: -63.8638077, 43.2302704, -61.8681450, 41.9122696, -105.7760620, 105.0984192
9: -49.5417900, 49.5968857, -48.0327950, 48.1215897, -97.6633759, 97.6296844

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 119

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3336048, upper bound: 107.3336048
time: 11.87 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3336048, upper bound: 107.3336048
time: 11.49 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -51.1037636, 40.2284279, -54.0346527, 42.5305176, -93.6342545, 94.2630768
1: -42.3624306, 36.0231743, -44.7858047, 38.0262260, -80.3886414, 80.8089752
2: -53.5631638, 32.5150604, -56.5906982, 34.2739792, -87.8371353, 89.1057510
3: -62.7546883, 29.6783867, -66.3619537, 31.2705040, -94.0251923, 96.0403290
4: -55.5695305, 42.5097809, -58.7910118, 44.9144173, -100.4839478, 101.3007965
5: -47.9975357, 36.9284096, -50.7273216, 38.9909668, -86.9884949, 87.6557312
6: -45.5012741, 47.0339737, -48.0667267, 49.7107277, -95.2119904, 95.1006851
7: -53.8413277, 36.7723732, -56.9172134, 38.6826096, -92.5239334, 93.6895905
8: -58.6167641, 39.7782135, -61.9059067, 42.0038109, -100.6205597, 101.6841202
9: -45.5904198, 45.7279472, -48.1487312, 48.3015518, -93.8919525, 93.8766785

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3336048, upper bound: 107.3336048
time: 8.16 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3336048, upper bound: 107.3336048
time: 11.12 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -55.8785019, 43.9305115, -55.4533615, 43.5631256, -99.4416199, 99.3838654
1: -46.3291168, 39.2976151, -45.9522514, 39.0272293, -85.3563461, 85.2498627
2: -58.6304131, 35.6569977, -58.2908936, 35.6041222, -94.2345123, 93.9478912
3: -68.5108566, 32.4085579, -67.9119186, 32.3178139, -100.8286667, 100.3204803
4: -60.7483749, 46.3964882, -60.2234650, 46.0333633, -106.7817383, 106.6199493
5: -52.4560318, 40.3734322, -52.0753822, 40.1334343, -92.5894623, 92.4487915
6: -49.8098335, 51.2815895, -49.5322342, 50.7984772, -100.6083069, 100.8138275
7: -58.7223549, 40.4061050, -58.1627274, 40.4947624, -99.2171173, 98.5688095
8: -64.1853943, 43.4780693, -63.8638077, 43.2302704, -107.4156647, 107.3418655
9: -49.8385544, 49.9704895, -49.5417900, 49.5968857, -99.4354401, 99.5122833

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 84

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3324445, upper bound: 107.3324206
time: 11.69 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3324445, upper bound: 107.3324206
time: 11.67 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -56.2051353, 44.2379265, -51.1037636, 40.2284279, -96.4335632, 95.3416901
1: -46.6010132, 39.5023651, -42.3624306, 36.0231743, -82.6241913, 81.8647766
2: -58.8257828, 35.5553894, -53.5631638, 32.5150604, -91.3408432, 89.1185532
3: -69.0375366, 32.4143372, -62.7546883, 29.6783867, -98.7159119, 95.1690216
4: -61.1810303, 46.6940041, -55.5695305, 42.5097809, -103.6908112, 102.2635345
5: -52.7497406, 40.5151939, -47.9975357, 36.9284096, -89.6781464, 88.5127258
6: -49.9580879, 51.7085075, -45.5012741, 47.0339737, -96.9920502, 97.2097778
7: -59.2162018, 40.0691986, -53.8413277, 36.7723732, -95.9885712, 93.9105225
8: -64.3242950, 43.6354141, -58.6167641, 39.7782135, -104.1025085, 102.2521667
9: -50.0260086, 50.2250938, -45.5904198, 45.7279472, -95.7539520, 95.8155060

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3324445, upper bound: 107.3324206
time: 13.04 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3324445, upper bound: 107.3324206
time: 8.66 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -55.4456253, 43.5571709, -55.8785019, 43.9305115, -99.3761368, 99.4356689
1: -45.9455986, 39.0219116, -46.3291168, 39.2976151, -85.2432098, 85.3510284
2: -58.2821999, 35.5986366, -58.6304131, 35.6569977, -93.9391937, 94.2290497
3: -67.9026794, 32.3128281, -68.5108566, 32.4085579, -100.3112259, 100.8236847
4: -60.2148056, 46.0270958, -60.7483749, 46.3964882, -106.6112976, 106.7754669
5: -52.0680275, 40.1276093, -52.4560318, 40.3734322, -92.4414520, 92.5836411
6: -49.5249252, 50.7918282, -49.8098335, 51.2815895, -100.8065186, 100.6016617
7: -58.1548462, 40.4877586, -58.7223549, 40.4061050, -98.5609283, 99.2101135
8: -63.8538971, 43.2241364, -64.1853943, 43.4780693, -107.3319626, 107.4095230
9: -49.5344658, 49.5898590, -49.8385544, 49.9704895, -99.5049591, 99.4284058

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 84

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3324206, upper bound: 107.3324445
time: 11.93 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3324206, upper bound: 107.3324445
time: 11.30 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -51.0967560, 40.2231064, -56.2051353, 44.2379265, -95.3346863, 96.4282379
1: -42.3564682, 36.0183983, -46.6010132, 39.5023651, -81.8588257, 82.6194077
2: -53.5553703, 32.5101357, -58.8257828, 35.5553894, -89.1107635, 91.3359222
3: -62.7462845, 29.6739197, -69.0375366, 32.4143372, -95.1606140, 98.7114487
4: -55.5617485, 42.5041504, -61.1810303, 46.6940041, -102.2557449, 103.6851807
5: -47.9909515, 36.9231377, -52.7497406, 40.5151939, -88.5061493, 89.6728821
6: -45.4947205, 47.0280113, -49.9580879, 51.7085075, -97.2032166, 96.9860916
7: -53.8342743, 36.7660751, -59.2162018, 40.0691986, -93.9034729, 95.9822769
8: -58.6078033, 39.7728424, -64.3242950, 43.6354141, -102.2432022, 104.0971298
9: -45.5838013, 45.7216377, -50.0260086, 50.2250938, -95.8088989, 95.7476501

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3324206, upper bound: 107.3324445
time: 12.29 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3324206, upper bound: 107.3324445
time: 12.43 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -57.5390282, 45.2035751, -55.8785019, 43.9305115, -101.4695435, 101.0820694
1: -47.6957817, 40.4419022, -46.3291168, 39.2976151, -86.9933777, 86.7710114
2: -60.4326210, 36.8314285, -58.6304131, 35.6569977, -96.0896149, 95.4618225
3: -70.4769516, 33.4156151, -68.5108566, 32.4085579, -102.8854980, 101.9264679
4: -62.5254135, 47.7418213, -60.7483749, 46.3964882, -108.9219055, 108.4901962
5: -54.0111237, 41.5995140, -52.4560318, 40.3734322, -94.3845520, 94.0555420
6: -51.3478508, 52.7182426, -49.8098335, 51.2815895, -102.6294403, 102.5280762
7: -60.3766823, 41.8249054, -58.7223549, 40.4061050, -100.7827682, 100.5472565
8: -66.1814728, 44.7947159, -64.1853943, 43.4780693, -109.6595459, 108.9801102
9: -51.3495941, 51.4457016, -49.8385544, 49.9704895, -101.3200760, 101.2842560

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3323532, upper bound: 107.3323532
time: 10.34 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3323532, upper bound: 107.3323532
time: 20.96 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -53.2817001, 41.9414177, -56.2051353, 44.2379265, -97.5196228, 98.1465530
1: -44.1837845, 37.5041504, -46.6010132, 39.5023651, -83.6861267, 84.1051636
2: -55.8071289, 33.8083878, -58.8257828, 35.5553894, -91.3625183, 92.6341705
3: -65.4367676, 30.8302536, -69.0375366, 32.4143372, -97.8511047, 99.8677826
4: -57.9710884, 44.2957726, -61.1810303, 46.6940041, -104.6650772, 105.4767990
5: -50.0250549, 38.4612732, -52.7497406, 40.5151939, -90.5402527, 91.2110138
6: -47.4047012, 49.0347786, -49.9580879, 51.7085075, -99.1131897, 98.9928665
7: -56.1456947, 38.1771088, -59.2162018, 40.0691986, -96.2148895, 97.3933105
8: -61.0464821, 41.4164619, -64.3242950, 43.6354141, -104.6818924, 105.7407532
9: -47.4787521, 47.6607704, -50.0260086, 50.2250938, -97.7038422, 97.6867828

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3323532, upper bound: 107.3323532
time: 11.33 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3323532, upper bound: 107.3323532
time: 10.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.36 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.36
Output dim: 7, lower bound: -107.3336048, upper bound: 107.3336048
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.36
Output dim: 7, lower bound: -107.3336048, upper bound: 107.3336048
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.36
Output dim: 7, lower bound: -107.3336048, upper bound: 107.3336048
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.36
Output dim: 7, lower bound: -107.3336048, upper bound: 107.3336048
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.36
Output dim: 7, lower bound: -107.3324445, upper bound: 107.3324206
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.36
Output dim: 7, lower bound: -107.3324445, upper bound: 107.3324206
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.36
Output dim: 7, lower bound: -107.3324445, upper bound: 107.3324206
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.36
Output dim: 7, lower bound: -107.3324445, upper bound: 107.3324206
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.36
Output dim: 7, lower bound: -107.3324206, upper bound: 107.3324445
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.36
Output dim: 7, lower bound: -107.3324206, upper bound: 107.3324445
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.36
Output dim: 7, lower bound: -107.3324206, upper bound: 107.3324445
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.36
Output dim: 7, lower bound: -107.3324206, upper bound: 107.3324445
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 23.36
Output dim: 7, lower bound: -107.3323532, upper bound: 107.3323532
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 23.36
Output dim: 7, lower bound: -107.3323532, upper bound: 107.3323532
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.36
Output dim: 7, lower bound: -107.3323532, upper bound: 107.3323532
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.36
Output dim: 7, lower bound: -107.3323532, upper bound: 107.3323532

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -53.7935791, 42.2900391, -53.7935791, 42.2900391, -96.0836182, 96.0836182
1: -44.5852928, 37.8822327, -44.5852928, 37.8822327, -82.4675293, 82.4675293
2: -56.4886551, 34.4299355, -56.4886551, 34.4299355, -90.9185944, 90.9185944
3: -65.9439926, 31.3121815, -65.9439926, 31.3121815, -97.2561722, 97.2561722
4: -58.4458122, 44.6872253, -58.4458122, 44.6872253, -103.1330414, 103.1330414
5: -50.5194397, 38.9095802, -50.5194397, 38.9095802, -89.4290161, 89.4290161
6: -47.9954147, 49.3605423, -47.9954147, 49.3605423, -97.3559494, 97.3559494
7: -56.5085754, 39.0789452, -56.5085754, 39.0789452, -95.5875092, 95.5875015
8: -61.8681450, 41.9122696, -61.8681450, 41.9122696, -103.7804108, 103.7804108
9: -48.0327950, 48.1215897, -48.0327950, 48.1215897, -96.1543884, 96.1543884

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3155295, upper bound: 107.3155985
time: 14.28 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2

### Relational analysis result of IS_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3141030, upper bound: 107.3140664
time: 10.42 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -54.0346527, 42.5305176, -53.7935791, 42.2900391, -96.3246918, 96.3240814
1: -44.7858047, 38.0262260, -44.5852928, 37.8822327, -82.6680374, 82.6115112
2: -56.5906982, 34.2739792, -56.4886551, 34.4299355, -91.0206299, 90.7626343
3: -66.3619537, 31.2705040, -65.9439926, 31.3121815, -97.6741333, 97.2144928
4: -58.7910118, 44.9144173, -58.4458122, 44.6872253, -103.4782410, 103.3602295
5: -50.7273216, 38.9909668, -50.5194397, 38.9095802, -89.6368942, 89.5104065
6: -48.0667267, 49.7107277, -47.9954147, 49.3605423, -97.4272690, 97.7061234
7: -56.9172134, 38.6826096, -56.5085754, 39.0789452, -95.9961548, 95.1911850
8: -61.9059067, 42.0038109, -61.8681450, 41.9122696, -103.8181610, 103.8719482
9: -48.1487312, 48.3015518, -48.0327950, 48.1215897, -96.2703171, 96.3343506

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3155295, upper bound: 107.3155985
time: 12.32 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3141030, upper bound: 107.3140664
time: 10.51 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -53.6693649, 42.1941299, -54.0346527, 42.5305176, -96.1998672, 96.2287827
1: -44.4759178, 37.7986946, -44.7858047, 38.0262260, -82.5021439, 82.5844879
2: -56.3553391, 34.3425751, -56.5906982, 34.2739792, -90.6293182, 90.9332733
3: -65.7939148, 31.2457619, -66.3619537, 31.2705040, -97.0644226, 97.6077118
4: -58.3074951, 44.5966454, -58.7910118, 44.9144173, -103.2219009, 103.3876572
5: -50.4011002, 38.8259773, -50.7273216, 38.9909668, -89.3920593, 89.5532990
6: -47.8866539, 49.2572556, -48.0667267, 49.7107277, -97.5973740, 97.3239822
7: -56.3826904, 38.9743042, -56.9172134, 38.6826096, -95.0653000, 95.8915176
8: -61.7325363, 41.8113861, -61.9059067, 42.0038109, -103.7363281, 103.7172852
9: -47.9307976, 48.0123711, -48.1487312, 48.3015518, -96.2323456, 96.1611023

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3150181, upper bound: 107.3149138
time: 14.81 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3136731, upper bound: 107.3136731
time: 14.39 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -54.0239906, 42.5234451, -54.0346527, 42.5305176, -96.5544891, 96.5580978
1: -44.7766800, 38.0205574, -44.7858047, 38.0262260, -82.8028870, 82.8063660
2: -56.5813065, 34.2671585, -56.5906982, 34.2739792, -90.8552856, 90.8578491
3: -66.3502121, 31.2657280, -66.3619537, 31.2705040, -97.6207123, 97.6276703
4: -58.7808113, 44.9077415, -58.7910118, 44.9144173, -103.6952286, 103.6987534
5: -50.7171555, 38.9850159, -50.7273216, 38.9909668, -89.7081223, 89.7123260
6: -48.0579376, 49.7026367, -48.0667267, 49.7107277, -97.7686539, 97.7693634
7: -56.9082222, 38.6753998, -56.9172134, 38.6826096, -95.5908203, 95.5926132
8: -61.8945160, 41.9947777, -61.9059067, 42.0038109, -103.8983078, 103.9006729
9: -48.1414948, 48.2938538, -48.1487312, 48.3015518, -96.4430389, 96.4425812

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3150181, upper bound: 107.3149138
time: 14.40 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3136731, upper bound: 107.3136731
time: 11.07 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -55.8785019, 43.9305115, -53.7935791, 42.2900391, -98.1685410, 97.7240906
1: -46.3291168, 39.2976151, -44.5852928, 37.8822327, -84.2113495, 83.8829041
2: -58.6304131, 35.6569977, -56.4886551, 34.4299355, -93.0603485, 92.1456528
3: -68.5108566, 32.4085579, -65.9439926, 31.3121815, -99.8230362, 98.3525467
4: -60.7483749, 46.3964882, -58.4458122, 44.6872253, -105.4356003, 104.8423004
5: -52.4560318, 40.3734322, -50.5194397, 38.9095802, -91.3656158, 90.8928680
6: -49.8098335, 51.2815895, -47.9954147, 49.3605423, -99.1703720, 99.2770081
7: -58.7223549, 40.4061050, -56.5085754, 39.0789452, -97.8013000, 96.9146652
8: -64.1853943, 43.4780693, -61.8681450, 41.9122696, -106.0976562, 105.3462143
9: -49.8385544, 49.9704895, -48.0327950, 48.1215897, -97.9601440, 98.0032806

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3138040, upper bound: 107.3138806
time: 11.28 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3127324, upper bound: 107.3127474
time: 13.91 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -55.8785019, 43.9305115, -54.0346527, 42.5305176, -98.4089966, 97.9651642
1: -46.3291168, 39.2976151, -44.7858047, 38.0262260, -84.3553467, 84.0834122
2: -58.6304131, 35.6569977, -56.5906982, 34.2739792, -92.9043808, 92.2476959
3: -68.5108566, 32.4085579, -66.3619537, 31.2705040, -99.7813568, 98.7705078
4: -60.7483749, 46.3964882, -58.7910118, 44.9144173, -105.6627960, 105.1875000
5: -52.4560318, 40.3734322, -50.7273216, 38.9909668, -91.4469986, 91.1007538
6: -49.8098335, 51.2815895, -48.0667267, 49.7107277, -99.5205536, 99.3483124
7: -58.7223549, 40.4061050, -56.9172134, 38.6826096, -97.4049683, 97.3233185
8: -64.1853943, 43.4780693, -61.9059067, 42.0038109, -106.1891937, 105.3839645
9: -49.8385544, 49.9704895, -48.1487312, 48.3015518, -98.1401062, 98.1192169

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 86

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3141521, upper bound: 107.3140910
time: 11.68 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3127324, upper bound: 107.3127475
time: 11.85 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -56.2051353, 44.2379265, -53.6693649, 42.1941299, -98.3992615, 97.9072876
1: -46.6010132, 39.5023651, -44.4759178, 37.7986946, -84.3997040, 83.9782715
2: -58.8257828, 35.5553894, -56.3553391, 34.3425751, -93.1683578, 91.9107285
3: -69.0375366, 32.4143372, -65.7939148, 31.2457619, -100.2832947, 98.2082443
4: -61.1810303, 46.6940041, -58.3074951, 44.5966454, -105.7776794, 105.0014801
5: -52.7497406, 40.5151939, -50.4011002, 38.8259773, -91.5757141, 90.9162903
6: -49.9580879, 51.7085075, -47.8866539, 49.2572556, -99.2153397, 99.5951538
7: -59.2162018, 40.0691986, -56.3826904, 38.9743042, -98.1905060, 96.4518890
8: -64.3242950, 43.6354141, -61.7325363, 41.8113861, -106.1356735, 105.3679276
9: -50.0260086, 50.2250938, -47.9307976, 48.0123711, -98.0383759, 98.1558914

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3136234, upper bound: 107.3136902
time: 13.40 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3122890, upper bound: 107.3122846
time: 11.13 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -56.2051353, 44.2379265, -54.0239906, 42.5234451, -98.7285767, 98.2619171
1: -46.6010132, 39.5023651, -44.7766800, 38.0205574, -84.6215668, 84.2790070
2: -58.8257828, 35.5553894, -56.5813065, 34.2671585, -93.0929413, 92.1366959
3: -69.0375366, 32.4143372, -66.3502121, 31.2657280, -100.3032608, 98.7645340
4: -61.1810303, 46.6940041, -58.7808113, 44.9077415, -106.0887680, 105.4747925
5: -52.7497406, 40.5151939, -50.7171555, 38.9850159, -91.7347488, 91.2323456
6: -49.9580879, 51.7085075, -48.0579376, 49.7026367, -99.6607208, 99.7664413
7: -59.2162018, 40.0691986, -56.9082222, 38.6753998, -97.8916016, 96.9774170
8: -64.3242950, 43.6354141, -61.8945160, 41.9947777, -106.3190613, 105.5299225
9: -50.0260086, 50.2250938, -48.1414948, 48.2938538, -98.3198624, 98.3665924

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3136234, upper bound: 107.3136902
time: 12.14 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3122890, upper bound: 107.3122846
time: 9.21 seconds

## BFS IS instance: IS_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -53.7861786, 42.2843552, -55.8785019, 43.9305115, -97.7166748, 98.1628571
1: -44.5789223, 37.8771515, -46.3291168, 39.2976151, -83.8765259, 84.2062683
2: -56.4803886, 34.4246941, -58.6304131, 35.6569977, -92.1373901, 93.0550995
3: -65.9351273, 31.3074493, -68.5108566, 32.4085579, -98.3436813, 99.8183060
4: -58.4375458, 44.6812363, -60.7483749, 46.3964882, -104.8340302, 105.4296112
5: -50.5124130, 38.9040070, -52.4560318, 40.3734322, -90.8858337, 91.3600388
6: -47.9884567, 49.3542099, -49.8098335, 51.2815895, -99.2700500, 99.1640396
7: -56.5010490, 39.0722847, -58.7223549, 40.4061050, -96.9071198, 97.7946396
8: -61.8587151, 41.9064941, -64.1853943, 43.4780693, -105.3367767, 106.0918884
9: -48.0258026, 48.1148834, -49.8385544, 49.9704895, -97.9962921, 97.9534378

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 186
type: A, layer: 1, pos: 186
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 68

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3138806, upper bound: 107.3138040
time: 14.39 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -107.3127475, upper bound: 107.3127324
time: 10.17 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.10 seconds
IS_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3155295, upper bound: 107.3155985
IS_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3141030, upper bound: 107.3140664
IS_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3155295, upper bound: 107.3155985
IS_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3141030, upper bound: 107.3140664
IS_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3150181, upper bound: 107.3149138
IS_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3136731, upper bound: 107.3136731
IS_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3150181, upper bound: 107.3149138
IS_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3136731, upper bound: 107.3136731
IS_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3138040, upper bound: 107.3138806
IS_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3127324, upper bound: 107.3127474
IS_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3141521, upper bound: 107.3140910
IS_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3127324, upper bound: 107.3127475
IS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3136234, upper bound: 107.3136902
IS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3122890, upper bound: 107.3122846
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3136234, upper bound: 107.3136902
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3122890, upper bound: 107.3122846
IS_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3138806, upper bound: 107.3138040
IS_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 26.10
Output dim: 7, lower bound: -107.3127475, upper bound: 107.3127324
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 26.10
Output dim: 7, lower bound: -107.3324206, upper bound: 107.3324445
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 26.10
Output dim: 7, lower bound: -107.3324206, upper bound: 107.3324445
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 26.10
Output dim: 7, lower bound: -107.3324206, upper bound: 107.3324445
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 26.10
Output dim: 7, lower bound: -107.3323532, upper bound: 107.3323532
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 26.10
Output dim: 7, lower bound: -107.3323532, upper bound: 107.3323532
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 26.10
Output dim: 7, lower bound: -107.3323532, upper bound: 107.3323532
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 26.10
Output dim: 7, lower bound: -107.3323532, upper bound: 107.3323532
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=120.15695190429688
rel_dist={7: [-107.34537841147622, 107.3453783970549]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1812.26 seconds
