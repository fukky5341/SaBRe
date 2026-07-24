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
execution time: IAR + LP analysis = 1.27 + 8.88 = 10.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -70.0817842, upper bound: 70.0817842


# Binary Search by BASE starts (time budget: 2689.84 seconds, max iter: 100)

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
Binary search time: 41.32 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2648.53 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0746555, upper bound: 70.0743540
time: 8.38 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0782883, upper bound: 70.0782883
time: 8.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.37 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.37
Output dim: 8, lower bound: -70.0746555, upper bound: 70.0743540
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.37
Output dim: 8, lower bound: -70.0782883, upper bound: 70.0782883

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -34.1880875, 27.3090477, -39.1711235, 31.2058411, -65.3939285, 66.4801559
1: -30.3645611, 24.3566647, -34.5777321, 27.7967625, -58.1613121, 58.9343948
2: -38.7411499, 24.0230026, -44.2465820, 27.4944324, -66.2355804, 68.2695847
3: -41.7636871, 20.4316597, -47.6548538, 23.4343376, -65.1980286, 68.0865173
4: -39.2750244, 27.7809067, -44.5984268, 31.9342289, -71.2092514, 72.3793182
5: -33.9946480, 26.4270020, -38.7571793, 30.0994282, -64.0940781, 65.1841736
6: -31.2516422, 30.8566246, -35.8540726, 35.2531281, -66.5047607, 66.7106934
7: -34.4891586, 32.2834358, -39.4727669, 36.4495201, -70.9386749, 71.7562027
8: -47.2628746, 22.9783707, -53.3833313, 27.0515957, -74.3144684, 76.3616867
9: -30.4555626, 30.6065769, -34.9177551, 35.0364532, -65.4920197, 65.5243301

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0729472, upper bound: 70.0729472
time: 8.09 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0729472, upper bound: 70.0743540
time: 6.44 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -36.9001198, 29.4238663, -40.2857857, 32.0787888, -68.9789047, 69.7096329
1: -32.6607742, 26.2213898, -35.5273781, 28.5705376, -61.2313080, 61.7487564
2: -41.7305946, 25.9101830, -45.4768066, 28.2778454, -70.0084305, 71.3869934
3: -44.9490280, 22.0794220, -48.9736557, 24.1043568, -69.0533752, 71.0530777
4: -42.1676102, 30.0317726, -45.8082924, 32.8718872, -75.0394974, 75.8400650
5: -36.5722961, 28.4246750, -39.8284340, 30.9265594, -67.4988556, 68.2531128
6: -33.7645187, 33.2581062, -36.8836212, 36.2383995, -70.0028992, 70.1417236
7: -37.1974792, 34.5521011, -40.5858040, 37.3845253, -74.5820007, 75.1379089
8: -50.5960922, 25.1990318, -54.7539024, 27.9678822, -78.5639725, 79.9529343
9: -32.8803673, 33.0069580, -35.9190674, 36.0372391, -68.9176025, 68.9260254

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0743540, upper bound: 70.0746555
time: 9.29 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0743540, upper bound: 70.0782883
time: 7.85 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 18.45 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 18.45
Output dim: 8, lower bound: -70.0729472, upper bound: 70.0729472
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 18.45
Output dim: 8, lower bound: -70.0729472, upper bound: 70.0743540
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 18.45
Output dim: 8, lower bound: -70.0743540, upper bound: 70.0746555
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 18.45
Output dim: 8, lower bound: -70.0743540, upper bound: 70.0782883

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -34.1880875, 27.3090477, -34.1880875, 27.3090477, -61.4971313, 61.4971161
1: -30.3645611, 24.3566647, -30.3645611, 24.3566647, -54.7212219, 54.7212181
2: -38.7411499, 24.0230026, -38.7411499, 24.0230026, -62.7641525, 62.7641525
3: -41.7636871, 20.4316597, -41.7636871, 20.4316597, -62.1953354, 62.1953468
4: -39.2750244, 27.7809067, -39.2750244, 27.7809067, -67.0559082, 67.0559235
5: -33.9946480, 26.4270020, -33.9946480, 26.4270020, -60.4216461, 60.4216461
6: -31.2516422, 30.8566246, -31.2516422, 30.8566246, -62.1082497, 62.1082497
7: -34.4891586, 32.2834358, -34.4891586, 32.2834358, -66.7725983, 66.7725983
8: -47.2628746, 22.9783707, -47.2628746, 22.9783707, -70.2412262, 70.2412262
9: -30.4555626, 30.6065769, -30.4555626, 30.6065769, -61.0621414, 61.0621338

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0722610, upper bound: 70.0722907
time: 8.98 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0720229, upper bound: 70.0720226
time: 7.84 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -34.1880875, 27.3090477, -36.9001198, 29.4238663, -63.6119537, 64.2091675
1: -30.3645611, 24.3566647, -32.6607742, 26.2213898, -56.5859528, 57.0174332
2: -38.7411499, 24.0230026, -41.7305946, 25.9101830, -64.6513367, 65.7536011
3: -41.7636871, 20.4316597, -44.9490280, 22.0794220, -63.8431015, 65.3806763
4: -39.2750244, 27.7809067, -42.1676102, 30.0317726, -69.3067856, 69.9485168
5: -33.9946480, 26.4270020, -36.5722961, 28.4246750, -62.4193230, 62.9992943
6: -31.2516422, 30.8566246, -33.7645187, 33.2581062, -64.5097504, 64.6211243
7: -34.4891586, 32.2834358, -37.1974792, 34.5521011, -69.0412598, 69.4809113
8: -47.2628746, 22.9783707, -50.5960922, 25.1990318, -72.4618988, 73.5744629
9: -30.4555626, 30.6065769, -32.8803673, 33.0069580, -63.4625168, 63.4869385

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0722610, upper bound: 70.0738170
time: 7.90 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0720229, upper bound: 70.0736142
time: 7.44 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -36.9001198, 29.4238663, -34.1880875, 27.3090477, -64.2091675, 63.6119461
1: -32.6607742, 26.2213898, -30.3645611, 24.3566647, -57.0174370, 56.5859489
2: -41.7305946, 25.9101830, -38.7411499, 24.0230026, -65.7535934, 64.6513290
3: -44.9490280, 22.0794220, -41.7636871, 20.4316597, -65.3806763, 63.8431091
4: -42.1676102, 30.0317726, -39.2750244, 27.7809067, -69.9485168, 69.3067932
5: -36.5722961, 28.4246750, -33.9946480, 26.4270020, -62.9992981, 62.4193230
6: -33.7645187, 33.2581062, -31.2516422, 30.8566246, -64.6211243, 64.5097504
7: -37.1974792, 34.5521011, -34.4891586, 32.2834358, -69.4809113, 69.0412598
8: -50.5960922, 25.1990318, -47.2628746, 22.9783707, -73.5744553, 72.4619064
9: -32.8803673, 33.0069580, -30.4555626, 30.6065769, -63.4869385, 63.4625206

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0737527, upper bound: 70.0740267
time: 6.44 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0736142, upper bound: 70.0738224
time: 7.75 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -36.9001198, 29.4238663, -36.9001198, 29.4238663, -66.3239746, 66.3239746
1: -32.6607742, 26.2213898, -32.6607742, 26.2213898, -58.8821640, 58.8821640
2: -41.7305946, 25.9101830, -41.7305946, 25.9101830, -67.6407776, 67.6407776
3: -44.9490280, 22.0794220, -44.9490280, 22.0794220, -67.0284424, 67.0284424
4: -42.1676102, 30.0317726, -42.1676102, 30.0317726, -72.1993866, 72.1993866
5: -36.5722961, 28.4246750, -36.5722961, 28.4246750, -64.9969711, 64.9969635
6: -33.7645187, 33.2581062, -33.7645187, 33.2581062, -67.0226212, 67.0226212
7: -37.1974792, 34.5521011, -37.1974792, 34.5521011, -71.7495804, 71.7495804
8: -50.5960922, 25.1990318, -50.5960922, 25.1990318, -75.7951202, 75.7951202
9: -32.8803673, 33.0069580, -32.8803673, 33.0069580, -65.8873215, 65.8873215

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0737527, upper bound: 70.0740267
time: 15.03 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0736142, upper bound: 70.0774666
time: 9.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.98 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.98
Output dim: 8, lower bound: -70.0722610, upper bound: 70.0722907
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.98
Output dim: 8, lower bound: -70.0720229, upper bound: 70.0720226
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.98
Output dim: 8, lower bound: -70.0722610, upper bound: 70.0738170
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.98
Output dim: 8, lower bound: -70.0720229, upper bound: 70.0736142
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.98
Output dim: 8, lower bound: -70.0737527, upper bound: 70.0740267
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.98
Output dim: 8, lower bound: -70.0736142, upper bound: 70.0738224
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.98
Output dim: 8, lower bound: -70.0737527, upper bound: 70.0740267
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.98
Output dim: 8, lower bound: -70.0736142, upper bound: 70.0774666

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -30.0785542, 24.0577011, -34.0822525, 27.2256126, -57.3041687, 58.1399536
1: -26.7405510, 21.4916096, -30.2713318, 24.2827301, -51.0232697, 51.7629395
2: -34.0965042, 21.1978455, -38.6213989, 23.9505005, -58.0470047, 59.8192368
3: -36.8384705, 18.0923119, -41.6365280, 20.3714771, -57.2099457, 59.7288399
4: -34.6125946, 24.4671402, -39.1551514, 27.6952515, -62.3078461, 63.6222916
5: -29.9094257, 23.2789402, -33.8893089, 26.3461876, -56.2556000, 57.1682510
6: -27.5399399, 27.2095356, -31.1562405, 30.7627583, -58.3026962, 58.3657722
7: -30.3673801, 28.4995689, -34.3829002, 32.1860809, -62.5534592, 62.8824615
8: -41.8193970, 20.1704121, -47.1225739, 22.9066143, -64.7260132, 67.2929840
9: -26.7874374, 26.9873600, -30.3611393, 30.5133095, -57.3007469, 57.3484993

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0720229, upper bound: 70.0720226
time: 7.37 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0720229, upper bound: 70.0720229
time: 8.60 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -32.0359230, 25.6056728, -33.6134300, 26.8557491, -58.8916702, 59.2191010
1: -28.4984684, 22.8607845, -29.8616009, 23.9562473, -52.4547119, 52.7223854
2: -36.3343506, 22.5545254, -38.0938110, 23.6284370, -59.9627876, 60.6483269
3: -39.2184143, 19.2202187, -41.0757980, 20.1012592, -59.3196716, 60.2960167
4: -36.8619766, 26.0420036, -38.6289558, 27.3155308, -64.1775055, 64.6709595
5: -31.8523369, 24.7737255, -33.4250870, 25.9888802, -57.8412094, 58.1988144
6: -29.3191128, 28.9648285, -30.7321091, 30.3473358, -59.6664505, 59.6969376
7: -32.3630257, 30.3131275, -33.9146996, 31.7598534, -64.1228790, 64.2278290
8: -44.4566917, 21.4998207, -46.5070229, 22.5802345, -67.0369263, 68.0068283
9: -28.5338268, 28.7276840, -29.9432373, 30.0995026, -58.6333275, 58.6709213

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0705140, upper bound: 70.0703883
time: 7.43 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0698025, upper bound: 70.0698025
time: 8.64 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -30.0785542, 24.0577011, -36.7895241, 29.3359337, -59.4144859, 60.8472214
1: -26.7405510, 21.4916096, -32.5634995, 26.1440544, -52.8845978, 54.0551071
2: -34.0965042, 21.1978455, -41.6057434, 25.8340607, -59.9305611, 62.8035889
3: -36.8384705, 18.0923119, -44.8154106, 22.0164051, -58.8548737, 62.9077225
4: -34.6125946, 24.4671402, -42.0421982, 29.9416027, -64.5541840, 66.5093384
5: -29.9094257, 23.2789402, -36.4612961, 28.3403091, -58.2497330, 59.7402344
6: -27.5399399, 27.2095356, -33.6647072, 33.1600914, -60.7000313, 60.8742371
7: -30.3673801, 28.4995689, -37.0861473, 34.4505730, -64.8179474, 65.5857162
8: -41.8193970, 20.1704121, -50.4504166, 25.1225357, -66.9419327, 70.6208191
9: -26.7874374, 26.9873600, -32.7813950, 32.9090500, -59.6964874, 59.7687531

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0738224, upper bound: 70.0736142
time: 8.80 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0738224, upper bound: 70.0736142
time: 8.68 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -32.0359230, 25.6056728, -36.2768936, 28.9291496, -60.9650650, 61.8825645
1: -28.4984684, 22.8607845, -32.1159096, 25.7863503, -54.2848053, 54.9766922
2: -36.3343506, 22.5545254, -41.0300598, 25.4813461, -61.8156853, 63.5845871
3: -39.2184143, 19.2202187, -44.1983490, 21.7218628, -60.9402771, 63.4185638
4: -36.8619766, 26.0420036, -41.4663467, 29.5232430, -66.3852158, 67.5083466
5: -31.8523369, 24.7737255, -35.9499893, 27.9509239, -59.8032608, 60.7237053
6: -29.3191128, 28.9648285, -33.2012100, 32.7063141, -62.0254211, 62.1660271
7: -32.3630257, 30.3131275, -36.5728035, 33.9854507, -66.3484802, 66.8859253
8: -44.4566917, 21.4998207, -49.7809525, 24.7606888, -69.2173767, 71.2807770
9: -28.5338268, 28.7276840, -32.3233376, 32.4549561, -60.9887772, 61.0510101

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0722391, upper bound: 70.0718649
time: 10.60 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0719408, upper bound: 70.0715945
time: 8.36 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -32.6069832, 26.0311852, -34.0822525, 27.2256126, -59.8325958, 60.1134377
1: -28.9047337, 23.2237034, -30.2713318, 24.2827301, -53.1874619, 53.4950333
2: -36.8972130, 22.9628830, -38.6213989, 23.9505005, -60.8477097, 61.5842743
3: -39.7977638, 19.6267662, -41.6365280, 20.3714771, -60.1692429, 61.2632866
4: -37.3360367, 26.5413742, -39.1551514, 27.6952515, -65.0312881, 65.6965256
5: -32.2924271, 25.1567688, -33.8893089, 26.3461876, -58.6386147, 59.0460739
6: -29.8887978, 29.4567661, -31.1562405, 30.7627583, -60.6515503, 60.6130028
7: -32.8990364, 30.6413994, -34.3829002, 32.1860809, -65.0851135, 65.0242996
8: -44.9631386, 22.2107620, -47.1225739, 22.9066143, -67.8697510, 69.3333359
9: -29.0490665, 29.2235146, -30.3611393, 30.5133095, -59.5623703, 59.5846405

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0736142, upper bound: 70.0738224
time: 9.04 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0736142, upper bound: 70.0738221
time: 8.82 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -34.6373672, 27.6285076, -33.6134300, 26.8557491, -61.4931107, 61.2419319
1: -30.7062569, 24.6442986, -29.8616009, 23.9562473, -54.6624985, 54.5058975
2: -39.2128754, 24.3687477, -38.0938110, 23.6284370, -62.8413010, 62.4625511
3: -42.2727966, 20.7985001, -41.0757980, 20.1012592, -62.3740540, 61.8742981
4: -39.6551323, 28.1848068, -38.6289558, 27.3155308, -66.9706650, 66.8137665
5: -34.3008766, 26.6993904, -33.4250870, 25.9888802, -60.2897568, 60.1244774
6: -31.7356853, 31.2765274, -30.7321091, 30.3473358, -62.0830116, 62.0086365
7: -34.9739952, 32.5010986, -33.9146996, 31.7598534, -66.7338486, 66.4158020
8: -47.6804352, 23.6211033, -46.5070229, 22.5802345, -70.2606659, 70.1281204
9: -30.8700676, 31.0265427, -29.9432373, 30.0995026, -60.9695625, 60.9697800

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0725762, upper bound: 70.0728313
time: 10.19 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0715945, upper bound: 70.0719408
time: 8.84 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -32.6069832, 26.0311852, -36.7895241, 29.3359337, -61.9429169, 62.8207016
1: -28.9047337, 23.2237034, -32.5634995, 26.1440544, -55.0487900, 55.7871971
2: -36.8972130, 22.9628830, -41.6057434, 25.8340607, -62.7312737, 64.5686264
3: -39.7977638, 19.6267662, -44.8154106, 22.0164051, -61.8141708, 64.4421768
4: -37.3360367, 26.5413742, -42.0421982, 29.9416027, -67.2776337, 68.5835724
5: -32.2924271, 25.1567688, -36.4612961, 28.3403091, -60.6327362, 61.6180649
6: -29.8887978, 29.4567661, -33.6647072, 33.1600914, -63.0488815, 63.1214752
7: -32.8990364, 30.6413994, -37.0861473, 34.4505730, -67.3496094, 67.7275467
8: -44.9631386, 22.2107620, -50.4504166, 25.1225357, -70.0856781, 72.6611633
9: -29.0490665, 29.2235146, -32.7813950, 32.9090500, -61.9581032, 62.0048981

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0774700, upper bound: 70.0774666
time: 7.87 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0774700, upper bound: 70.0774666
time: 7.29 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -34.6373672, 27.6285076, -36.2768936, 28.9291496, -63.5664978, 63.9053993
1: -30.7062569, 24.6442986, -32.1159096, 25.7863503, -56.4926071, 56.7602081
2: -39.2128754, 24.3687477, -41.0300598, 25.4813461, -64.6942139, 65.3988037
3: -42.2727966, 20.7985001, -44.1983490, 21.7218628, -63.9946594, 64.9968338
4: -39.6551323, 28.1848068, -41.4663467, 29.5232430, -69.1783752, 69.6511536
5: -34.3008766, 26.6993904, -35.9499893, 27.9509239, -62.2518005, 62.6493797
6: -31.7356853, 31.2765274, -33.2012100, 32.7063141, -64.4419937, 64.4777374
7: -34.9739952, 32.5010986, -36.5728035, 33.9854507, -68.9594421, 69.0738983
8: -47.6804352, 23.6211033, -49.7809525, 24.7606888, -72.4411240, 73.4020538
9: -30.8700676, 31.0265427, -32.3233376, 32.4549561, -63.3250046, 63.3498764

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0765769, upper bound: 70.0765805
time: 9.79 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0761910, upper bound: 70.0761907
time: 5.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 16.29 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 8, lower bound: -70.0720229, upper bound: 70.0720226
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 8, lower bound: -70.0720229, upper bound: 70.0720229
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 8, lower bound: -70.0705140, upper bound: 70.0703883
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 8, lower bound: -70.0698025, upper bound: 70.0698025
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 8, lower bound: -70.0738224, upper bound: 70.0736142
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 8, lower bound: -70.0738224, upper bound: 70.0736142
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 8, lower bound: -70.0722391, upper bound: 70.0718649
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 8, lower bound: -70.0719408, upper bound: 70.0715945
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 8, lower bound: -70.0736142, upper bound: 70.0738224
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 8, lower bound: -70.0736142, upper bound: 70.0738221
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 8, lower bound: -70.0725762, upper bound: 70.0728313
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 8, lower bound: -70.0715945, upper bound: 70.0719408
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 8, lower bound: -70.0774700, upper bound: 70.0774666
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 8, lower bound: -70.0774700, upper bound: 70.0774666
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 8, lower bound: -70.0765769, upper bound: 70.0765805
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.29
Output dim: 8, lower bound: -70.0761910, upper bound: 70.0761907

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -30.0785542, 24.0577011, -30.0785542, 24.0577011, -54.1362534, 54.1362534
1: -26.7405510, 21.4916096, -26.7405510, 21.4916096, -48.2321587, 48.2321587
2: -34.0965042, 21.1978455, -34.0965042, 21.1978455, -55.2943497, 55.2943497
3: -36.8384705, 18.0923119, -36.8384705, 18.0923119, -54.9307823, 54.9307823
4: -34.6125946, 24.4671402, -34.6125946, 24.4671402, -59.0797272, 59.0797348
5: -29.9094257, 23.2789402, -29.9094257, 23.2789402, -53.1883621, 53.1883621
6: -27.5399399, 27.2095356, -27.5399399, 27.2095356, -54.7494736, 54.7494736
7: -30.3673801, 28.4995689, -30.3673801, 28.4995689, -58.8669434, 58.8669472
8: -41.8193970, 20.1704121, -41.8193970, 20.1704121, -61.9898071, 61.9898071
9: -26.7874374, 26.9873600, -26.7874374, 26.9873600, -53.7747955, 53.7747955

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0706249, upper bound: 70.0707694
time: 7.23 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0699965, upper bound: 70.0700706
time: 9.45 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -30.0785542, 24.0577011, -32.0359230, 25.6056728, -55.6842270, 56.0936241
1: -26.7405510, 21.4916096, -28.4984684, 22.8607845, -49.6013336, 49.9900780
2: -34.0965042, 21.1978455, -36.3343506, 22.5545254, -56.6510277, 57.5321960
3: -36.8384705, 18.0923119, -39.2184143, 19.2202187, -56.0586891, 57.3107262
4: -34.6125946, 24.4671402, -36.8619766, 26.0420036, -60.6545982, 61.3291168
5: -29.9094257, 23.2789402, -31.8523369, 24.7737255, -54.6831512, 55.1312714
6: -27.5399399, 27.2095356, -29.3191128, 28.9648285, -56.5047684, 56.5286407
7: -30.3673801, 28.4995689, -32.3630257, 30.3131275, -60.6804962, 60.8625908
8: -41.8193970, 20.1704121, -44.4566917, 21.4998207, -63.3192177, 64.6271057
9: -26.7874374, 26.9873600, -28.5338268, 28.7276840, -55.5151215, 55.5211830

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0706249, upper bound: 70.0707691
time: 7.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0699965, upper bound: 70.0700706
time: 8.52 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -32.0359230, 25.6056728, -31.8355827, 25.4701653, -57.5060883, 57.4412537
1: -28.4984684, 22.8607845, -28.3406181, 22.7350178, -51.2334862, 51.2014008
2: -36.3343506, 22.5545254, -36.1349411, 22.3854008, -58.7197495, 58.6894684
3: -39.2184143, 19.2202187, -38.9822388, 19.0582256, -58.2766418, 58.2024574
4: -36.8619766, 26.0420036, -36.7014122, 25.8744049, -62.7363777, 62.7434158
5: -31.8523369, 24.7737255, -31.7238293, 24.6728745, -56.5252075, 56.4975471
6: -29.3191128, 28.9648285, -29.1060905, 28.7738876, -58.0929985, 58.0709152
7: -32.3630257, 30.3131275, -32.1490555, 30.2255535, -62.5885773, 62.4621811
8: -44.4566917, 21.4998207, -44.3016434, 21.2265129, -65.6832047, 65.8014526
9: -28.5338268, 28.7276840, -28.3591766, 28.5400124, -57.0738373, 57.0868607

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0698025, upper bound: 70.0698025
time: 7.13 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0698025, upper bound: 70.0698025
time: 8.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -31.7132568, 25.3537865, -35.0620041, 28.0136051, -59.7268600, 60.4157829
1: -28.2217350, 22.6395817, -31.2004871, 24.9966888, -53.2184219, 53.8400688
2: -35.9786034, 22.3286095, -39.7975807, 24.6062851, -60.5848885, 62.1261902
3: -38.8384705, 19.0319061, -42.9046516, 20.9619350, -59.8004074, 61.9365578
4: -36.5117302, 25.7814331, -40.3794937, 28.5095062, -65.0212173, 66.1609268
5: -31.5443287, 24.5344276, -34.9041862, 27.1225815, -58.6669083, 59.4386139
6: -29.0239887, 28.6797752, -32.0454941, 31.6516228, -60.6756096, 60.7252655
7: -32.0424576, 30.0338364, -35.4105263, 33.1952896, -65.2377472, 65.4443665
8: -44.0561218, 21.2552834, -48.6299438, 23.4563026, -67.5124207, 69.8852234
9: -28.2466011, 28.4455566, -31.2402649, 31.4155769, -59.6621780, 59.6858177

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0698025, upper bound: 70.0698025
time: 7.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0698025, upper bound: 70.0698022
time: 10.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -30.0785542, 24.0577011, -32.6069832, 26.0311852, -56.1097374, 56.6646843
1: -26.7405510, 21.4916096, -28.9047337, 23.2237034, -49.9642487, 50.3963432
2: -34.0965042, 21.1978455, -36.8972130, 22.9628830, -57.0593834, 58.0950584
3: -36.8384705, 18.0923119, -39.7977638, 19.6267662, -56.4652367, 57.8900757
4: -34.6125946, 24.4671402, -37.3360367, 26.5413742, -61.1539650, 61.8031769
5: -29.9094257, 23.2789402, -32.2924271, 25.1567688, -55.0661888, 55.5713654
6: -27.5399399, 27.2095356, -29.8887978, 29.4567661, -56.9967041, 57.0983238
7: -30.3673801, 28.4995689, -32.8990364, 30.6413994, -61.0087814, 61.3985939
8: -41.8193970, 20.1704121, -44.9631386, 22.2107620, -64.0301590, 65.1335449
9: -26.7874374, 26.9873600, -29.0490665, 29.2235146, -56.0109520, 56.0364265

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0730410, upper bound: 70.0728541
time: 19.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0721674, upper bound: 70.0718539
time: 11.19 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -30.0785542, 24.0577011, -34.6373672, 27.6285076, -57.7070580, 58.6950684
1: -26.7405510, 21.4916096, -30.7062569, 24.6442986, -51.3848495, 52.1978683
2: -34.0965042, 21.1978455, -39.2128754, 24.3687477, -58.4652481, 60.4107208
3: -36.8384705, 18.0923119, -42.2727966, 20.7985001, -57.6369667, 60.3651085
4: -34.6125946, 24.4671402, -39.6551323, 28.1848068, -62.7974014, 64.1222687
5: -29.9094257, 23.2789402, -34.3008766, 26.6993904, -56.6088181, 57.5798187
6: -27.5399399, 27.2095356, -31.7356853, 31.2765274, -58.8164673, 58.9452019
7: -30.3673801, 28.4995689, -34.9739952, 32.5010986, -62.8684769, 63.4735641
8: -41.8193970, 20.1704121, -47.6804352, 23.6211033, -65.4404984, 67.8508453
9: -26.7874374, 26.9873600, -30.8700676, 31.0265427, -57.8139801, 57.8574295

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0730410, upper bound: 70.0728541
time: 10.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0721674, upper bound: 70.0718539
time: 9.40 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -32.0359230, 25.6056728, -34.2984009, 27.3913155, -59.4272308, 59.9040680
1: -28.4984684, 22.8607845, -30.4365215, 24.4225559, -52.9210205, 53.2973061
2: -36.3343506, 22.5545254, -38.8536873, 24.1035099, -60.4378586, 61.4082069
3: -39.2184143, 19.2202187, -41.8500595, 20.5604687, -59.7788811, 61.0702782
4: -36.8619766, 26.0420036, -39.3264809, 27.9058990, -64.7678680, 65.3684845
5: -31.8523369, 24.7737255, -34.0531044, 26.4909134, -58.3432465, 58.8268280
6: -29.3191128, 28.9648285, -31.3939342, 30.9573631, -60.2764740, 60.3587532
7: -32.3630257, 30.3131275, -34.6025543, 32.2909698, -64.6539917, 64.9156723
8: -44.4566917, 21.4998207, -47.3307762, 23.2295780, -67.6862717, 68.8305893
9: -28.5338268, 28.7276840, -30.5540009, 30.7195320, -59.2533455, 59.2816849

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0719408, upper bound: 70.0715945
time: 8.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0719408, upper bound: 70.0715945
time: 9.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -31.7132568, 25.3537865, -37.8138962, 30.1688995, -61.8821564, 63.1676750
1: -28.2217350, 22.6395817, -33.5485802, 26.8947411, -55.1164742, 56.1881523
2: -35.9786034, 22.3286095, -42.8491554, 26.5302162, -62.5088120, 65.1777573
3: -38.8384705, 19.0319061, -46.1395721, 22.6276703, -61.4661369, 65.1714706
4: -36.5117302, 25.7814331, -43.3369942, 30.7903271, -67.3020554, 69.1184235
5: -31.5443287, 24.5344276, -37.5269012, 29.1612625, -60.7055893, 62.0613213
6: -29.0239887, 28.6797752, -34.6017914, 34.0970955, -63.1210709, 63.2815628
7: -32.0424576, 30.0338364, -38.1570969, 35.5192947, -67.5617523, 68.1909332
8: -44.0561218, 21.2552834, -52.0452271, 25.6855888, -69.7417068, 73.3005066
9: -28.2466011, 28.4455566, -33.7011223, 33.8616295, -62.1082153, 62.1466599

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0719408, upper bound: 70.0715945
time: 8.11 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0719408, upper bound: 70.0715945
time: 8.33 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -32.6069832, 26.0311852, -30.0785542, 24.0577011, -56.6646843, 56.1097374
1: -28.9047337, 23.2237034, -26.7405510, 21.4916096, -50.3963432, 49.9642487
2: -36.8972130, 22.9628830, -34.0965042, 21.1978455, -58.0950584, 57.0593872
3: -39.7977638, 19.6267662, -36.8384705, 18.0923119, -57.8900757, 56.4652328
4: -37.3360367, 26.5413742, -34.6125946, 24.4671402, -61.8031769, 61.1539612
5: -32.2924271, 25.1567688, -29.9094257, 23.2789402, -55.5713654, 55.0661850
6: -29.8887978, 29.4567661, -27.5399399, 27.2095356, -57.0983200, 56.9967041
7: -32.8990364, 30.6413994, -30.3673801, 28.4995689, -61.3985977, 61.0087814
8: -44.9631386, 22.2107620, -41.8193970, 20.1704121, -65.1335449, 64.0301590
9: -29.0490665, 29.2235146, -26.7874374, 26.9873600, -56.0364227, 56.0109406

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0720165, upper bound: 70.0724500
time: 8.00 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0717749, upper bound: 70.0721302
time: 16.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -32.6069832, 26.0311852, -32.0359230, 25.6056728, -58.2126541, 58.0671005
1: -28.9047337, 23.2237034, -28.4984684, 22.8607845, -51.7655182, 51.7221718
2: -36.8972130, 22.9628830, -36.3343506, 22.5545254, -59.4517365, 59.2972336
3: -39.7977638, 19.6267662, -39.2184143, 19.2202187, -59.0179825, 58.8451767
4: -37.3360367, 26.5413742, -36.8619766, 26.0420036, -63.3780403, 63.4033432
5: -32.2924271, 25.1567688, -31.8523369, 24.7737255, -57.0661545, 57.0091019
6: -29.8887978, 29.4567661, -29.3191128, 28.9648285, -58.8536148, 58.7758713
7: -32.8990364, 30.6413994, -32.3630257, 30.3131275, -63.2121582, 63.0044250
8: -44.9631386, 22.2107620, -44.4566917, 21.4998207, -66.4629593, 66.6674500
9: -29.0490665, 29.2235146, -28.5338268, 28.7276840, -57.7767410, 57.7573242

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0720165, upper bound: 70.0724503
time: 7.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0717749, upper bound: 70.0721302
time: 9.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -34.6373672, 27.6285076, -31.8355827, 25.4701653, -60.1075287, 59.4640884
1: -30.7062569, 24.6442986, -28.3406181, 22.7350178, -53.4412766, 52.9849167
2: -39.2128754, 24.3687477, -36.1349411, 22.3854008, -61.5982742, 60.5036850
3: -42.2727966, 20.7985001, -38.9822388, 19.0582256, -61.3310242, 59.7807388
4: -39.6551323, 28.1848068, -36.7014122, 25.8744049, -65.5295258, 64.8862152
5: -34.3008766, 26.6993904, -31.7238293, 24.6728745, -58.9737511, 58.4232140
6: -31.7356853, 31.2765274, -29.1060905, 28.7738876, -60.5095634, 60.3826180
7: -34.9739952, 32.5010986, -32.1490555, 30.2255535, -65.1995468, 64.6501541
8: -47.6804352, 23.6211033, -44.3016434, 21.2265129, -68.9069519, 67.9227448
9: -30.8700676, 31.0265427, -28.3591766, 28.5400124, -59.4100800, 59.3857193

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0715945, upper bound: 70.0719408
time: 10.29 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0715945, upper bound: 70.0719408
time: 10.08 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -34.2748108, 27.3476963, -35.0620041, 28.0136051, -62.2884140, 62.4096947
1: -30.3995838, 24.3944359, -31.2004871, 24.9966888, -55.3962708, 55.5949249
2: -38.8144455, 24.1161842, -39.7975807, 24.6062851, -63.4207306, 63.9137650
3: -41.8409500, 20.5863934, -42.9046516, 20.9619350, -62.8028870, 63.4910431
4: -39.2633514, 27.8880577, -40.3794937, 28.5095062, -67.7728500, 68.2675476
5: -33.9548683, 26.4307327, -34.9041862, 27.1225815, -61.0774460, 61.3349190
6: -31.4042339, 30.9567528, -32.0454941, 31.6516228, -63.0558434, 63.0022354
7: -34.6116409, 32.1911850, -35.4105263, 33.1952896, -67.8069153, 67.6017151
8: -47.2313080, 23.3413105, -48.6299438, 23.4563026, -70.6876068, 71.9712524
9: -30.5437260, 30.7103043, -31.2402649, 31.4155769, -61.9592972, 61.9505692

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0690986, upper bound: 70.0697142
time: 6.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0682707, upper bound: 70.0686090
time: 9.18 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -32.6069832, 26.0311852, -32.6069832, 26.0311852, -58.6381645, 58.6381607
1: -28.9047337, 23.2237034, -28.9047337, 23.2237034, -52.1284370, 52.1284370
2: -36.8972130, 22.9628830, -36.8972130, 22.9628830, -59.8600960, 59.8600922
3: -39.7977638, 19.6267662, -39.7977638, 19.6267662, -59.4245300, 59.4245300
4: -37.3360367, 26.5413742, -37.3360367, 26.5413742, -63.8774109, 63.8774033
5: -32.2924271, 25.1567688, -32.2924271, 25.1567688, -57.4491920, 57.4491959
6: -29.8887978, 29.4567661, -29.8887978, 29.4567661, -59.3455544, 59.3455582
7: -32.8990364, 30.6413994, -32.8990364, 30.6413994, -63.5404358, 63.5404358
8: -44.9631386, 22.2107620, -44.9631386, 22.2107620, -67.1739044, 67.1739044
9: -29.0490665, 29.2235146, -29.0490665, 29.2235146, -58.2725677, 58.2725563

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0767932, upper bound: 70.0768023
time: 10.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0763987, upper bound: 70.0763996
time: 7.97 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -32.6069832, 26.0311852, -34.6373672, 27.6285076, -60.2354889, 60.6685410
1: -28.9047337, 23.2237034, -30.7062569, 24.6442986, -53.5490341, 53.9299622
2: -36.8972130, 22.9628830, -39.2128754, 24.3687477, -61.2659531, 62.1757507
3: -39.7977638, 19.6267662, -42.2727966, 20.7985001, -60.5962563, 61.8995628
4: -37.3360367, 26.5413742, -39.6551323, 28.1848068, -65.5208435, 66.1965027
5: -32.2924271, 25.1567688, -34.3008766, 26.6993904, -58.9918175, 59.4576454
6: -29.8887978, 29.4567661, -31.7356853, 31.2765274, -61.1653214, 61.1924362
7: -32.8990364, 30.6413994, -34.9739952, 32.5010986, -65.4001312, 65.6153946
8: -44.9631386, 22.2107620, -47.6804352, 23.6211033, -68.5842438, 69.8911972
9: -29.0490665, 29.2235146, -30.8700676, 31.0265427, -60.0756035, 60.0935783

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0767932, upper bound: 70.0768023
time: 11.10 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0763987, upper bound: 70.0763996
time: 8.20 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -34.6373672, 27.6285076, -34.2984009, 27.3913155, -62.0286560, 61.9269028
1: -30.7062569, 24.6442986, -30.4365215, 24.4225559, -55.1288109, 55.0808182
2: -39.2128754, 24.3687477, -38.8536873, 24.1035099, -63.3163757, 63.2224236
3: -42.2727966, 20.7985001, -41.8500595, 20.5604687, -62.8332634, 62.6485519
4: -39.6551323, 28.1848068, -39.3264809, 27.9058990, -67.5610199, 67.5112915
5: -34.3008766, 26.6993904, -34.0531044, 26.4909134, -60.7917900, 60.7524948
6: -31.7356853, 31.2765274, -31.3939342, 30.9573631, -62.6930389, 62.6704597
7: -34.9739952, 32.5010986, -34.6025543, 32.2909698, -67.2649689, 67.1036530
8: -47.6804352, 23.6211033, -47.3307762, 23.2295780, -70.9100113, 70.9518814
9: -30.8700676, 31.0265427, -30.5540009, 30.7195320, -61.5895882, 61.5805435

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0761910, upper bound: 70.0761907
time: 9.17 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0761910, upper bound: 70.0761907
time: 9.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -34.2748108, 27.3476963, -37.8138962, 30.1688995, -64.4437103, 65.1615906
1: -30.3995838, 24.3944359, -33.5485802, 26.8947411, -57.2943268, 57.9430046
2: -38.8144455, 24.1161842, -42.8491554, 26.5302162, -65.3446579, 66.9653397
3: -41.8409500, 20.5863934, -46.1395721, 22.6276703, -64.4686203, 66.7259598
4: -39.2633514, 27.8880577, -43.3369942, 30.7903271, -70.0536804, 71.2250519
5: -33.9548683, 26.4307327, -37.5269012, 29.1612625, -63.1161308, 63.9576302
6: -31.4042339, 30.9567528, -34.6017914, 34.0970955, -65.5013199, 65.5585327
7: -34.6116409, 32.1911850, -38.1570969, 35.5192947, -70.1309357, 70.3482819
8: -47.2313080, 23.3413105, -52.0452271, 25.6855888, -72.9168930, 75.3865356
9: -30.5437260, 30.7103043, -33.7011223, 33.8616295, -64.4053345, 64.4114151

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0736583, upper bound: 70.0739773
time: 11.40 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0728196, upper bound: 70.0728253
time: 7.86 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 20.62 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0706249, upper bound: 70.0707694
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0699965, upper bound: 70.0700706
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0706249, upper bound: 70.0707691
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0699965, upper bound: 70.0700706
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0698025, upper bound: 70.0698025
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0698025, upper bound: 70.0698025
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0698025, upper bound: 70.0698025
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0698025, upper bound: 70.0698022
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0730410, upper bound: 70.0728541
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0721674, upper bound: 70.0718539
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0730410, upper bound: 70.0728541
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0721674, upper bound: 70.0718539
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0719408, upper bound: 70.0715945
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0719408, upper bound: 70.0715945
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0719408, upper bound: 70.0715945
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0719408, upper bound: 70.0715945
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0720165, upper bound: 70.0724500
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0717749, upper bound: 70.0721302
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0720165, upper bound: 70.0724503
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0717749, upper bound: 70.0721302
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0715945, upper bound: 70.0719408
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0715945, upper bound: 70.0719408
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0690986, upper bound: 70.0697142
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0682707, upper bound: 70.0686090
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0767932, upper bound: 70.0768023
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0763987, upper bound: 70.0763996
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0767932, upper bound: 70.0768023
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0763987, upper bound: 70.0763996
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0761910, upper bound: 70.0761907
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0761910, upper bound: 70.0761907
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0736583, upper bound: 70.0739773
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.62
Output dim: 8, lower bound: -70.0728196, upper bound: 70.0728253
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=82.72178649902344
rel_dist={8: [-70.0816870801838, 70.08168706258579]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0739132, upper bound: 70.0736846
time: 10.53 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0782441, upper bound: 70.0782441
time: 8.90 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.56 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.56
Output dim: 8, lower bound: -70.0739132, upper bound: 70.0736846
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.56
Output dim: 8, lower bound: -70.0782441, upper bound: 70.0782441

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -34.1880875, 27.3090477, -37.1319618, 29.6122017, -63.8002853, 64.4410019
1: -30.3645611, 24.3566647, -32.8415527, 26.3845901, -56.7491531, 57.1982193
2: -38.7411499, 24.0230026, -41.9946976, 26.0706749, -64.8118286, 66.0177002
3: -41.7636871, 20.4316597, -45.2287827, 22.2187576, -63.9824448, 65.6604385
4: -39.2750244, 27.7809067, -42.4084587, 30.2287292, -69.5037384, 70.1893616
5: -33.9946480, 26.4270020, -36.7961960, 28.5967503, -62.5914001, 63.2231827
6: -31.2516422, 30.8566246, -33.9788055, 33.4569778, -64.7086105, 64.8354111
7: -34.4891586, 32.2834358, -37.4365692, 34.7342033, -69.2233582, 69.7199936
8: -47.2628746, 22.9783707, -50.8689270, 25.4002190, -72.6630859, 73.8472977
9: -30.4555626, 30.6065769, -33.0952415, 33.2131500, -63.6687088, 63.7018127

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0731077, upper bound: 70.0729898
time: 9.91 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0729716, upper bound: 70.0728877
time: 9.67 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -36.9001198, 29.4238663, -39.0581207, 31.1173496, -68.0174713, 68.4819870
1: -32.6607742, 26.2213898, -34.4862556, 27.7176285, -60.3784027, 60.7076378
2: -41.7305946, 25.9101830, -44.1199493, 27.4170494, -69.1476440, 70.0301285
3: -44.9490280, 22.0794220, -47.5189667, 23.3663044, -68.3153305, 69.5983887
4: -42.1676102, 30.0317726, -44.4843559, 31.8402367, -74.0078430, 74.5161133
5: -36.5722961, 28.4246750, -38.6479301, 30.0181103, -66.5904083, 67.0726013
6: -33.7645187, 33.2581062, -35.7514992, 35.1557693, -68.9202881, 69.0096054
7: -37.1974792, 34.5521011, -39.3597221, 36.3600235, -73.5574951, 73.9118195
8: -50.5960922, 25.1990318, -53.2506790, 26.9573364, -77.5534286, 78.4496918
9: -32.8803673, 33.0069580, -34.8168755, 34.9355011, -67.8158569, 67.8238373

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0736846, upper bound: 70.0739129
time: 8.98 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0736846, upper bound: 70.0782441
time: 8.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.03 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 19.03
Output dim: 8, lower bound: -70.0731077, upper bound: 70.0729898
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 19.03
Output dim: 8, lower bound: -70.0729716, upper bound: 70.0728877
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 19.03
Output dim: 8, lower bound: -70.0736846, upper bound: 70.0739129
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 19.03
Output dim: 8, lower bound: -70.0736846, upper bound: 70.0782441

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -32.4583054, 25.9430618, -32.8073997, 26.1880627, -58.6463699, 58.7504616
1: -28.8386269, 23.1495209, -29.0608158, 23.3628922, -52.2015076, 52.2103348
2: -36.7838173, 22.8355522, -37.1251450, 23.1014729, -59.8852921, 59.9606857
3: -39.6876984, 19.4470253, -40.0330734, 19.7477551, -59.4354553, 59.4800949
4: -37.3149147, 26.3814812, -37.5410995, 26.7093811, -64.0242844, 63.9225769
5: -32.2725945, 25.1044712, -32.4788628, 25.3050423, -57.5776367, 57.5833359
6: -29.6905136, 29.3220787, -30.0751724, 29.6275177, -59.3180313, 59.3972511
7: -32.7520905, 30.6898880, -33.1042442, 30.7985458, -63.5506363, 63.7941322
8: -44.9682808, 21.8017902, -45.2023087, 22.3829803, -67.3512573, 67.0040970
9: -28.9113617, 29.0818729, -29.2337608, 29.3979511, -58.3093109, 58.3156319

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0717036, upper bound: 70.0715551
time: 9.84 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0711207, upper bound: 70.0708882
time: 9.49 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -32.4762421, 25.9567070, -34.8707428, 27.8129768, -60.2892075, 60.8274498
1: -28.8645363, 23.1646156, -30.8902321, 24.8074245, -53.6719589, 54.0548439
2: -36.8116455, 22.8461990, -39.4771805, 24.5311279, -61.3427620, 62.3233795
3: -39.7153587, 19.4485207, -42.5485954, 20.9387264, -60.6540833, 61.9971085
4: -37.3480682, 26.3951931, -39.8952904, 28.3822308, -65.7303009, 66.2904816
5: -32.2978325, 25.1203671, -34.5184708, 26.8731194, -59.1709518, 59.6388283
6: -29.7027225, 29.3388519, -31.9511051, 31.4770088, -61.1797333, 61.2899551
7: -32.7766876, 30.7197914, -35.2118759, 32.6846390, -65.4613266, 65.9316635
8: -45.0077438, 21.7923641, -47.9590645, 23.8209496, -68.8286896, 69.7514267
9: -28.9276428, 29.0954952, -31.0857372, 31.2308788, -60.1585083, 60.1812286

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0715923, upper bound: 70.0714891
time: 10.21 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0709924, upper bound: 70.0707833
time: 11.31 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -36.9001198, 29.4238663, -34.1880875, 27.3090477, -64.2091675, 63.6119461
1: -32.6607742, 26.2213898, -30.3645611, 24.3566647, -57.0174370, 56.5859489
2: -41.7305946, 25.9101830, -38.7411499, 24.0230026, -65.7535934, 64.6513290
3: -44.9490280, 22.0794220, -41.7636871, 20.4316597, -65.3806763, 63.8431091
4: -42.1676102, 30.0317726, -39.2750244, 27.7809067, -69.9485168, 69.3067932
5: -36.5722961, 28.4246750, -33.9946480, 26.4270020, -62.9992981, 62.4193230
6: -33.7645187, 33.2581062, -31.2516422, 30.8566246, -64.6211243, 64.5097504
7: -37.1974792, 34.5521011, -34.4891586, 32.2834358, -69.4809113, 69.0412598
8: -50.5960922, 25.1990318, -47.2628746, 22.9783707, -73.5744553, 72.4619064
9: -32.8803673, 33.0069580, -30.4555626, 30.6065769, -63.4869385, 63.4625206

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0729898, upper bound: 70.0731077
time: 8.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0728877, upper bound: 70.0729716
time: 10.99 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -36.9001198, 29.4238663, -36.9001198, 29.4238663, -66.3239746, 66.3239746
1: -32.6607742, 26.2213898, -32.6607742, 26.2213898, -58.8821640, 58.8821640
2: -41.7305946, 25.9101830, -41.7305946, 25.9101830, -67.6407776, 67.6407776
3: -44.9490280, 22.0794220, -44.9490280, 22.0794220, -67.0284424, 67.0284424
4: -42.1676102, 30.0317726, -42.1676102, 30.0317726, -72.1993866, 72.1993866
5: -36.5722961, 28.4246750, -36.5722961, 28.4246750, -64.9969711, 64.9969635
6: -33.7645187, 33.2581062, -33.7645187, 33.2581062, -67.0226212, 67.0226212
7: -37.1974792, 34.5521011, -37.1974792, 34.5521011, -71.7495804, 71.7495804
8: -50.5960922, 25.1990318, -50.5960922, 25.1990318, -75.7951202, 75.7951202
9: -32.8803673, 33.0069580, -32.8803673, 33.0069580, -65.8873215, 65.8873215

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0729898, upper bound: 70.0731074
time: 11.68 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0728877, upper bound: 70.0774452
time: 10.09 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.13 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.13
Output dim: 8, lower bound: -70.0717036, upper bound: 70.0715551
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.13
Output dim: 8, lower bound: -70.0711207, upper bound: 70.0708882
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.13
Output dim: 8, lower bound: -70.0715923, upper bound: 70.0714891
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.13
Output dim: 8, lower bound: -70.0709924, upper bound: 70.0707833
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.13
Output dim: 8, lower bound: -70.0729898, upper bound: 70.0731077
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.13
Output dim: 8, lower bound: -70.0728877, upper bound: 70.0729716
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.13
Output dim: 8, lower bound: -70.0729898, upper bound: 70.0731074
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.13
Output dim: 8, lower bound: -70.0728877, upper bound: 70.0774452

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -30.7060547, 24.5772152, -32.0977554, 25.6370926, -56.3431358, 56.6749725
1: -27.3385811, 21.9470119, -28.4577026, 22.8739967, -50.2125778, 50.4047050
2: -34.8540001, 21.6095657, -36.3438568, 22.6062298, -57.4602280, 57.9534225
3: -37.6239891, 18.4201164, -39.1912384, 19.3309269, -56.9549179, 57.6113548
4: -35.4103622, 24.9655418, -36.7717972, 26.1317863, -61.5421486, 61.7373390
5: -30.5996704, 23.8066406, -31.8013515, 24.7795277, -55.3791962, 55.6079826
6: -28.0896454, 27.7698631, -29.4247417, 28.9986324, -57.0882721, 57.1945953
7: -31.0111885, 29.1759529, -32.3941307, 30.1908512, -61.2020416, 61.5700836
8: -42.7906647, 20.4711437, -44.3208237, 21.8341007, -64.6247635, 64.7919693
9: -27.3517704, 27.5450516, -28.5959549, 28.7760429, -56.1278152, 56.1410065

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0689170, upper bound: 70.0685537
time: 11.59 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0683898, upper bound: 70.0682029
time: 11.06 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -33.8373260, 27.0481377, -31.8049545, 25.4077682, -59.2450943, 58.8530884
1: -30.1208153, 24.1411552, -28.2075481, 22.6725006, -52.7933044, 52.3486938
2: -38.4124947, 23.7699776, -36.0201492, 22.4012566, -60.8137512, 59.7901230
3: -41.4404831, 20.2612343, -38.8440895, 19.1616154, -60.6020966, 59.1053238
4: -38.9987106, 27.5157356, -36.4555130, 25.8956108, -64.8943176, 63.9712486
5: -33.6800003, 26.1909943, -31.5211163, 24.5615177, -58.2415161, 57.7121124
6: -30.9417648, 30.5670166, -29.1571579, 28.7417145, -59.6834793, 59.7241669
7: -34.1821289, 32.0745926, -32.1039352, 29.9368095, -64.1189346, 64.1785202
8: -47.0138741, 22.6214409, -43.9557838, 21.6145687, -68.6284256, 66.5772247
9: -30.1533375, 30.3374405, -28.3349609, 28.5206184, -58.6739578, 58.6724014

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0683994, upper bound: 70.0679737
time: 11.44 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0678051, upper bound: 70.0674974
time: 11.66 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -30.7266903, 24.5930500, -34.1418533, 27.2488670, -57.9755554, 58.7349014
1: -27.3667107, 21.9643173, -30.2738228, 24.3059597, -51.6726685, 52.2381363
2: -34.8849182, 21.6219597, -38.6763496, 24.0233631, -58.9082794, 60.2983017
3: -37.6553726, 18.4233189, -41.6837158, 20.5123692, -58.1677399, 60.1070328
4: -35.4465714, 24.9814472, -39.1065826, 27.7860985, -63.2326698, 64.0880280
5: -30.6275406, 23.8248425, -33.8233109, 26.3349495, -56.9624901, 57.6481438
6: -28.1042747, 27.7890434, -31.2853928, 30.8322487, -58.9365234, 59.0744362
7: -31.0386677, 29.2083721, -34.4828911, 32.0633430, -63.1020126, 63.6912613
8: -42.8340721, 20.4627972, -47.0577164, 23.2541161, -66.0881882, 67.5205154
9: -27.3706512, 27.5613918, -30.4299030, 30.5938454, -57.9644928, 57.9912796

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0688342, upper bound: 70.0684806
time: 9.74 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0682968, upper bound: 70.0681282
time: 10.61 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -33.8860283, 27.0859413, -33.8302879, 27.0059814, -60.8920097, 60.9162216
1: -30.1740227, 24.1784325, -30.0092812, 24.0901699, -54.2641869, 54.1877060
2: -38.4753227, 23.8013363, -38.3316879, 23.8052330, -62.2805519, 62.1330109
3: -41.5061417, 20.2805862, -41.3106384, 20.3329144, -61.8390579, 61.5912094
4: -39.0664406, 27.5547409, -38.7686310, 27.5313301, -66.5977554, 66.3233643
5: -33.7356606, 26.2304573, -33.5259514, 26.1017723, -59.8374329, 59.7564049
6: -30.9818192, 30.6114693, -31.0003033, 30.5586739, -61.5404930, 61.6117706
7: -34.2386551, 32.1331940, -34.1713562, 31.7946758, -66.0333328, 66.3045502
8: -47.0948257, 22.6316566, -46.6691132, 23.0180035, -70.1128235, 69.3007660
9: -30.1972008, 30.3787041, -30.1486778, 30.3222809, -60.5194778, 60.5273819

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0683003, upper bound: 70.0679027
time: 9.57 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0676944, upper bound: 70.0674262
time: 13.30 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -32.6069832, 26.0311852, -32.4583054, 25.9430618, -58.5500450, 58.4894867
1: -28.9047337, 23.2237034, -28.8386269, 23.1495209, -52.0542526, 52.0623245
2: -36.8972130, 22.9628830, -36.7838173, 22.8355522, -59.7327614, 59.7467003
3: -39.7977638, 19.6267662, -39.6876984, 19.4470253, -59.2447891, 59.3144608
4: -37.3360367, 26.5413742, -37.3149147, 26.3814812, -63.7175102, 63.8562889
5: -32.2924271, 25.1567688, -32.2725945, 25.1044712, -57.3968925, 57.4293518
6: -29.8887978, 29.4567661, -29.6905136, 29.3220787, -59.2108765, 59.1472702
7: -32.8990364, 30.6413994, -32.7520905, 30.6898880, -63.5889244, 63.3934898
8: -44.9631386, 22.2107620, -44.9682808, 21.8017902, -66.7649307, 67.1790466
9: -29.0490665, 29.2235146, -28.9113617, 29.0818729, -58.1309280, 58.1348610

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0715554, upper bound: 70.0717033
time: 10.14 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0708885, upper bound: 70.0711204
time: 10.98 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -34.6373672, 27.6285076, -32.4762421, 25.9567070, -60.5940704, 60.1047516
1: -30.7062569, 24.6442986, -28.8645363, 23.1646156, -53.8708725, 53.5088348
2: -39.2128754, 24.3687477, -36.8116455, 22.8461990, -62.0590630, 61.1803741
3: -42.2727966, 20.7985001, -39.7153587, 19.4485207, -61.7213173, 60.5138588
4: -39.6551323, 28.1848068, -37.3480682, 26.3951931, -66.0503235, 65.5328751
5: -34.3008766, 26.6993904, -32.2978325, 25.1203671, -59.4212418, 58.9972229
6: -31.7356853, 31.2765274, -29.7027225, 29.3388519, -61.0745354, 60.9792480
7: -34.9739952, 32.5010986, -32.7766876, 30.7197914, -65.6937866, 65.2777863
8: -47.6804352, 23.6211033, -45.0077438, 21.7923641, -69.4727936, 68.6288452
9: -30.8700676, 31.0265427, -28.9276428, 29.0954952, -59.9655609, 59.9541779

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0714891, upper bound: 70.0715923
time: 10.72 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0707833, upper bound: 70.0709924
time: 9.43 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -32.6069832, 26.0311852, -35.1253433, 28.0185890, -60.6255722, 61.1565285
1: -28.9047337, 23.2237034, -31.1053810, 24.9813309, -53.8860626, 54.3290825
2: -36.8972130, 22.9628830, -39.7293549, 24.6922626, -61.5894699, 62.6922340
3: -39.7977638, 19.6267662, -42.8119965, 21.0672874, -60.8650513, 62.4387627
4: -37.3360367, 26.5413742, -40.1647110, 28.5868721, -65.9229126, 66.7060852
5: -32.2924271, 25.1567688, -34.7981224, 27.0749550, -59.3673820, 59.9548912
6: -29.8887978, 29.4567661, -32.1638565, 31.6865692, -61.5753632, 61.6206169
7: -32.8990364, 30.6413994, -35.4169540, 32.9316673, -65.8307037, 66.0583496
8: -44.9631386, 22.2107620, -48.2640610, 23.9682312, -68.9313660, 70.4748230
9: -29.0490665, 29.2235146, -31.2963066, 31.4390602, -60.4881248, 60.5198135

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0765295, upper bound: 70.0765258
time: 9.17 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0762849, upper bound: 70.0762866
time: 10.65 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -34.6373672, 27.6285076, -35.0442200, 27.9553261, -62.5926895, 62.6727295
1: -30.7062569, 24.6442986, -31.0418472, 24.9264679, -55.6327209, 55.6861458
2: -39.2128754, 24.3687477, -39.6451149, 24.6360912, -63.8489609, 64.0138550
3: -42.2727966, 20.7985001, -42.7185135, 21.0136948, -63.2864914, 63.5170135
4: -39.6551323, 28.1848068, -40.0843277, 28.5186481, -68.1737671, 68.2691345
5: -34.3008766, 26.6993904, -34.7247238, 27.0166492, -61.3175278, 61.4241142
6: -31.7356853, 31.2765274, -32.0884705, 31.6155968, -63.3512726, 63.3649979
7: -34.9739952, 32.5010986, -35.3404350, 32.8693008, -67.8432922, 67.8415375
8: -47.6804352, 23.6211033, -48.1694489, 23.8938866, -71.5743179, 71.7905502
9: -30.8700676, 31.0265427, -31.2246628, 31.3657722, -62.2358398, 62.2511978

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0764004, upper bound: 70.0763840
time: 9.64 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0761659, upper bound: 70.0761658
time: 9.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 20.28 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 8, lower bound: -70.0689170, upper bound: 70.0685537
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 8, lower bound: -70.0683898, upper bound: 70.0682029
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 8, lower bound: -70.0683994, upper bound: 70.0679737
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 8, lower bound: -70.0678051, upper bound: 70.0674974
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 8, lower bound: -70.0688342, upper bound: 70.0684806
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 8, lower bound: -70.0682968, upper bound: 70.0681282
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 8, lower bound: -70.0683003, upper bound: 70.0679027
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 8, lower bound: -70.0676944, upper bound: 70.0674262
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 8, lower bound: -70.0715554, upper bound: 70.0717033
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 8, lower bound: -70.0708885, upper bound: 70.0711204
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 8, lower bound: -70.0714891, upper bound: 70.0715923
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 8, lower bound: -70.0707833, upper bound: 70.0709924
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 8, lower bound: -70.0765295, upper bound: 70.0765258
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 8, lower bound: -70.0762849, upper bound: 70.0762866
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 8, lower bound: -70.0764004, upper bound: 70.0763840
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 8, lower bound: -70.0761659, upper bound: 70.0761658

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -30.1912384, 24.1725216, -30.5643692, 24.4315414, -54.6227798, 54.7368889
1: -26.8916092, 21.5882854, -27.1322861, 21.8045673, -48.6961746, 48.7205734
2: -34.2817078, 21.2468681, -34.6406784, 21.5298119, -55.8115082, 55.8875465
3: -37.0078163, 18.1104126, -37.3594475, 18.4102364, -55.4180527, 55.4698486
4: -34.8415031, 24.5484676, -35.0876465, 24.8880959, -59.7295914, 59.6361160
5: -30.1035194, 23.4225025, -30.3231831, 23.6366806, -53.7401886, 53.7456818
6: -27.6211281, 27.3109837, -28.0286064, 27.6353569, -55.2564850, 55.3395844
7: -30.4941788, 28.7259445, -30.8588448, 28.8551216, -59.3492889, 59.5847893
8: -42.1318893, 20.0839291, -42.3664131, 20.6709404, -62.8028145, 62.4503403
9: -26.8953362, 27.0885525, -27.2352715, 27.4185085, -54.3138428, 54.3238220

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0600453, upper bound: 70.0590447
time: 11.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0689170, upper bound: 70.0685537
time: 11.46 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -29.9778709, 24.0047112, -33.7621384, 26.9666309, -56.9444923, 57.7668495
1: -26.7084389, 21.4372940, -29.9734135, 24.0273781, -50.7358093, 51.4107056
2: -34.0488434, 21.0951977, -38.3170738, 23.7490635, -57.7979050, 59.4122581
3: -36.7508736, 17.9736118, -41.2350197, 20.2489319, -56.9997978, 59.2086334
4: -34.6140556, 24.3719940, -38.7717896, 27.4878654, -62.1019211, 63.1437836
5: -29.9011803, 23.2659321, -33.4874802, 26.0741405, -55.9753189, 56.7533951
6: -27.4254818, 27.1198578, -30.9398670, 30.5076828, -57.9331627, 58.0597229
7: -30.2803059, 28.5495491, -34.1152420, 31.8307056, -62.1110115, 62.6647911
8: -41.8724251, 19.9079170, -46.7415314, 22.8715649, -64.7439880, 66.6494446
9: -26.7056160, 26.9006805, -30.1022224, 30.3171921, -57.0228081, 57.0029030

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0595613, upper bound: 70.0586449
time: 11.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0683890, upper bound: 70.0682029
time: 10.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -33.3123779, 26.6368637, -30.2821884, 24.2115479, -57.5239258, 56.9190445
1: -29.6665344, 23.7753830, -26.8909225, 21.6117249, -51.2782593, 50.6663055
2: -37.8301468, 23.4006996, -34.3284149, 21.3313274, -59.1614761, 57.7291145
3: -40.8139343, 19.9462891, -37.0265770, 18.2463303, -59.0602646, 56.9728661
4: -38.4190216, 27.0909004, -34.7809296, 24.6621952, -63.0812073, 61.8718185
5: -33.1740494, 25.8003082, -30.0542793, 23.4267654, -56.6008072, 55.8545837
6: -30.4640923, 30.0999565, -27.7717113, 27.3869705, -57.8510513, 57.8716621
7: -33.6556358, 31.6155319, -30.5783825, 28.6092548, -62.2648849, 62.1939125
8: -46.3435783, 22.2275181, -42.0130844, 20.4603424, -66.8039246, 64.2406006
9: -29.6876450, 29.8731079, -26.9840584, 27.1725063, -56.8601532, 56.8571663

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0618127, upper bound: 70.0614330
time: 12.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0601841, upper bound: 70.0593039
time: 12.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -33.0948563, 26.4660587, -33.4246521, 26.7005920, -59.7954369, 59.8907089
1: -29.4807549, 23.6212387, -29.6845055, 23.7908821, -53.2716370, 53.3057442
2: -37.5933075, 23.2463417, -37.9384079, 23.5105057, -61.1038055, 61.1847496
3: -40.5534592, 19.8070965, -40.8309784, 20.0500031, -60.6034622, 60.6380768
4: -38.1867714, 26.9111443, -38.4057388, 27.2111130, -65.3978882, 65.3168793
5: -32.9681282, 25.6403427, -33.1612473, 25.8171062, -58.7852325, 58.8015862
6: -30.2642212, 29.9054337, -30.6267662, 30.2100372, -60.4742508, 60.5321999
7: -33.4376259, 31.4355125, -33.7758560, 31.5362587, -64.9738693, 65.2113647
8: -46.0792046, 22.0480614, -46.3125153, 22.6160545, -68.6952591, 68.3605804
9: -29.4936409, 29.6814289, -29.7954369, 30.0188828, -59.5125237, 59.4768639

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0608363, upper bound: 70.0606586
time: 12.06 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0592801, upper bound: 70.0586353
time: 11.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -30.2150230, 24.1909885, -32.5465088, 25.9973221, -56.2123451, 56.7374954
1: -26.9225979, 21.6077518, -28.9020004, 23.1886044, -50.1112022, 50.5097504
2: -34.3160286, 21.2614441, -36.9055290, 22.9039536, -57.2199821, 58.1669693
3: -37.0429878, 18.1156425, -39.7694855, 19.5550270, -56.5980148, 57.8851280
4: -34.8811417, 24.5668659, -37.3544807, 26.4852581, -61.3664017, 61.9213486
5: -30.1343079, 23.4432011, -32.2867355, 25.1450729, -55.2793808, 55.7299271
6: -27.6386528, 27.3330593, -29.8313713, 29.4138298, -57.0524826, 57.1644249
7: -30.5248413, 28.7612152, -32.8818436, 30.6816273, -61.2064667, 61.6430588
8: -42.1792564, 20.0778675, -45.0341110, 22.0347385, -64.2139969, 65.1119766
9: -26.9170685, 27.1077538, -29.0058212, 29.1817169, -56.0987854, 56.1135674

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0600557, upper bound: 70.0590422
time: 10.20 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0688342, upper bound: 70.0684806
time: 10.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -29.9976044, 24.0199661, -35.7523766, 28.5366707, -58.5342751, 59.7723427
1: -26.7356873, 21.4538975, -31.7454319, 25.4207497, -52.1564331, 53.1993179
2: -34.0785103, 21.1068001, -40.5912704, 25.1272011, -59.2056961, 61.6980705
3: -36.7812805, 17.9765053, -43.6717339, 21.3925056, -58.1737785, 61.6482315
4: -34.6490326, 24.3871136, -41.0549812, 29.0990791, -63.7481079, 65.4420929
5: -29.9277592, 23.2836456, -35.4565163, 27.5902100, -57.5179672, 58.7401581
6: -27.4393425, 27.1381607, -32.7454910, 32.2988434, -59.7381859, 59.8836517
7: -30.3069096, 28.5809994, -36.1531296, 33.6606979, -63.9676056, 64.7341309
8: -41.9142990, 19.8989296, -49.4195442, 24.2415161, -66.1558151, 69.3184738
9: -26.7239609, 26.9161835, -31.8910408, 32.0861168, -58.8100777, 58.8072205

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0595624, upper bound: 70.0586465
time: 9.15 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0682968, upper bound: 70.0681279
time: 8.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -33.3623009, 26.6755447, -32.2494354, 25.7660656, -59.1283569, 58.9249802
1: -29.7205811, 23.8135605, -28.6484947, 22.9851017, -52.7056808, 52.4620552
2: -37.8940926, 23.4328651, -36.5774727, 22.6964798, -60.5905685, 60.0103378
3: -40.8811684, 19.9663181, -39.4168892, 19.3837414, -60.2649078, 59.3832016
4: -38.4880447, 27.1307793, -37.0324860, 26.2458057, -64.7338486, 64.1632614
5: -33.2308159, 25.8405972, -32.0027885, 24.9249992, -58.1558151, 57.8433723
6: -30.5050163, 30.1455116, -29.5597458, 29.1527214, -59.6577377, 59.7052574
7: -33.7131424, 31.6751041, -32.5853806, 30.4245033, -64.1376419, 64.2604828
8: -46.4262962, 22.2381973, -44.6628723, 21.8120708, -68.2383575, 66.9010696
9: -29.7324829, 29.9155159, -28.7401409, 28.9232006, -58.6556854, 58.6556549

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0616573, upper bound: 70.0612863
time: 10.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0600907, upper bound: 70.0592159
time: 10.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -33.1407928, 26.5015869, -35.4452896, 28.2964745, -61.4372559, 61.9468765
1: -29.5312824, 23.6566257, -31.4827461, 25.2079926, -54.7392654, 55.1393738
2: -37.6526871, 23.2756805, -40.2499504, 24.9124660, -62.5651398, 63.5256233
3: -40.6159630, 19.8247890, -43.3039780, 21.2154179, -61.8313675, 63.1287651
4: -38.2513657, 26.9478741, -40.7206688, 28.8482456, -67.0996094, 67.6685410
5: -33.0209770, 25.6776161, -35.1617088, 27.3606644, -60.3816299, 60.8393250
6: -30.3014603, 29.9474258, -32.4632950, 32.0278168, -62.3292770, 62.4107170
7: -33.4910774, 31.4913616, -35.8451576, 33.3938751, -66.8849487, 67.3365173
8: -46.1567688, 22.0559349, -49.0336266, 24.0103474, -70.1671143, 71.0895615
9: -29.5349808, 29.7203770, -31.6135635, 31.8173790, -61.3523521, 61.3339386

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0606407, upper bound: 70.0605145
time: 9.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0591492, upper bound: 70.0585308
time: 9.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -31.9118576, 25.4913120, -30.7060547, 24.5772152, -56.4890747, 56.1973648
1: -28.3128242, 22.7468262, -27.3385811, 21.9470119, -50.2598343, 50.0854073
2: -36.1326485, 22.4778843, -34.8540001, 21.6095657, -57.7422142, 57.3318863
3: -38.9753799, 19.2183189, -37.6239891, 18.4201164, -57.3954964, 56.8423080
4: -36.5804634, 25.9783363, -35.4103622, 24.9655418, -61.5459976, 61.3886986
5: -31.6298828, 24.6431961, -30.5996704, 23.8066406, -55.4365196, 55.2428665
6: -29.2522469, 28.8402939, -28.0896454, 27.7698631, -57.0221100, 56.9299393
7: -32.2041054, 30.0453377, -31.0111885, 29.1759529, -61.3800583, 61.0565262
8: -44.0996399, 21.6731300, -42.7906647, 20.4711437, -64.5707855, 64.4637909
9: -28.4259682, 28.6154900, -27.3517704, 27.5450516, -55.9710197, 55.9672623

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0685540, upper bound: 70.0689166
time: 9.18 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0682025, upper bound: 70.0683894
time: 9.96 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -31.6402321, 25.2790451, -33.8373260, 27.0481377, -58.6883698, 59.1163635
1: -28.0807571, 22.5601254, -30.1208153, 24.1411552, -52.2219124, 52.6809311
2: -35.8324509, 22.2880287, -38.4124947, 23.7699776, -59.6024246, 60.7005234
3: -38.6529579, 19.0615273, -41.4404831, 20.2612343, -58.9141922, 60.5020103
4: -36.2866058, 25.7598534, -38.9987106, 27.5157356, -63.8023415, 64.7585602
5: -31.3699398, 24.4410133, -33.6800003, 26.1909943, -57.5609360, 58.1210136
6: -29.0041847, 28.6023560, -30.9417648, 30.5670166, -59.5712013, 59.5441208
7: -31.9351540, 29.8093739, -34.1821289, 32.0745926, -64.0097504, 63.9914932
8: -43.7606812, 21.4706059, -47.0138741, 22.6214409, -66.3821259, 68.4844742
9: -28.1840992, 28.3787918, -30.1533375, 30.3374405, -58.5215378, 58.5321198

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0679737, upper bound: 70.0683997
time: 10.33 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0674977, upper bound: 70.0678051
time: 9.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -33.9101334, 27.0654926, -30.7266903, 24.5930500, -58.5031815, 57.7921829
1: -30.0904598, 24.1436672, -27.3667107, 21.9643173, -52.0547791, 51.5103760
2: -38.4134521, 23.8612976, -34.8849182, 21.6219597, -60.0354080, 58.7462082
3: -41.4083519, 20.3724365, -37.6553726, 18.4233189, -59.8316727, 58.0278015
4: -38.8669510, 27.5904007, -35.4465714, 24.9814472, -63.8483963, 63.0369720
5: -33.6074600, 26.1614532, -30.6275406, 23.8248425, -57.4322968, 56.7889938
6: -31.0706654, 30.6327229, -28.1042747, 27.7890434, -58.8597107, 58.7369995
7: -34.2455406, 31.8805084, -31.0386677, 29.2083721, -63.4539032, 62.9191704
8: -46.7794952, 23.0562820, -42.8340721, 20.4627972, -67.2422943, 65.8903503
9: -30.2151985, 30.3904667, -27.3706512, 27.5613918, -57.7765808, 57.7611008

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0684806, upper bound: 70.0688342
time: 8.03 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0681279, upper bound: 70.0682968
time: 11.32 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -33.6237335, 26.8430901, -33.8860283, 27.0859413, -60.7096710, 60.7291069
1: -29.8473320, 23.9467373, -30.1740227, 24.1784325, -54.0257645, 54.1207581
2: -38.0982437, 23.6619339, -38.4753227, 23.8013363, -61.8995819, 62.1372490
3: -41.0670547, 20.2075768, -41.5061417, 20.2805862, -61.3476372, 61.7137184
4: -38.5565720, 27.3582478, -39.0664406, 27.5547409, -66.1112976, 66.4246750
5: -33.3346786, 25.9490089, -33.7356606, 26.2304573, -59.5651360, 59.6846695
6: -30.8097229, 30.3816452, -30.9818192, 30.6114693, -61.4211884, 61.3634644
7: -33.9602203, 31.6340771, -34.2386551, 32.1331940, -66.0934143, 65.8727341
8: -46.4239502, 22.8404427, -47.0948257, 22.6316566, -69.0556030, 69.9352570
9: -29.9585876, 30.1417465, -30.1972008, 30.3787041, -60.3372917, 60.3389320

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0679027, upper bound: 70.0683003
time: 10.60 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0674262, upper bound: 70.0676944
time: 10.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.36 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0600453, upper bound: 70.0590447
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0689170, upper bound: 70.0685537
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0595613, upper bound: 70.0586449
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0683890, upper bound: 70.0682029
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0618127, upper bound: 70.0614330
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0601841, upper bound: 70.0593039
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0608363, upper bound: 70.0606586
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0592801, upper bound: 70.0586353
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0600557, upper bound: 70.0590422
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0688342, upper bound: 70.0684806
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0595624, upper bound: 70.0586465
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0682968, upper bound: 70.0681279
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0616573, upper bound: 70.0612863
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0600907, upper bound: 70.0592159
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0606407, upper bound: 70.0605145
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0591492, upper bound: 70.0585308
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0685540, upper bound: 70.0689166
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0682025, upper bound: 70.0683894
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0679737, upper bound: 70.0683997
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0674977, upper bound: 70.0678051
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0684806, upper bound: 70.0688342
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0681279, upper bound: 70.0682968
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0679027, upper bound: 70.0683003
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.36
Output dim: 8, lower bound: -70.0674262, upper bound: 70.0676944
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.36
Output dim: 8, lower bound: -70.0765295, upper bound: 70.0765258
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.36
Output dim: 8, lower bound: -70.0762849, upper bound: 70.0762866
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.36
Output dim: 8, lower bound: -70.0764004, upper bound: 70.0763840
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.36
Output dim: 8, lower bound: -70.0761659, upper bound: 70.0761658
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=82.72178649902344
rel_dist={8: [-70.08157121987698, 70.08157121987699]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0731573, upper bound: 70.0731028
time: 10.58 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0781063, upper bound: 70.0781063
time: 8.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.67 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.67
Output dim: 8, lower bound: -70.0731573, upper bound: 70.0731028
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.67
Output dim: 8, lower bound: -70.0781063, upper bound: 70.0781063

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -34.1880875, 27.3090477, -35.0697327, 27.9938812, -62.1819687, 62.3787727
1: -30.3645611, 24.3566647, -31.0921707, 24.9576340, -55.3221893, 55.4488335
2: -38.7411499, 24.0230026, -39.7071877, 24.6389885, -63.3801346, 63.7301903
3: -41.7636871, 20.4316597, -42.7797813, 20.9816570, -62.7453308, 63.2114334
4: -39.2750244, 27.7809067, -40.1872711, 28.5128517, -67.7878723, 67.9681702
5: -33.9946480, 26.4270020, -34.8123856, 27.0705395, -61.0651855, 61.2393875
6: -31.2516422, 30.8566246, -32.0777054, 31.6328926, -62.8845291, 62.9343033
7: -34.4891586, 32.2834358, -35.3670883, 32.9887276, -67.4778900, 67.6505280
8: -47.2628746, 22.9783707, -48.3026047, 23.7405357, -71.0033951, 71.2809601
9: -30.4555626, 30.6065769, -31.2467499, 31.3805904, -61.8361511, 61.8533249

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0722980, upper bound: 70.0722602
time: 9.12 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0722459, upper bound: 70.0722149
time: 8.97 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -36.9001198, 29.4238663, -37.7215118, 30.0688133, -66.9689331, 67.1453781
1: -32.6607742, 26.2213898, -33.3515930, 26.7896061, -59.4503784, 59.5729828
2: -41.7305946, 25.9101830, -42.6403198, 26.4833660, -68.2139511, 68.5504990
3: -44.9490280, 22.0794220, -45.9303055, 22.5675774, -67.5166016, 68.0097198
4: -42.1676102, 30.0317726, -43.0489922, 30.7198315, -72.8874435, 73.0807648
5: -36.5722961, 28.4246750, -37.3631134, 29.0307865, -65.6030807, 65.7877884
6: -33.7645187, 33.2581062, -34.5219460, 33.9792480, -67.7437668, 67.7800522
7: -37.1974792, 34.5521011, -38.0220680, 35.2420807, -72.4395599, 72.5741730
8: -50.5960922, 25.1990318, -51.6100197, 25.8655624, -76.4616547, 76.8090439
9: -32.8803673, 33.0069580, -33.6192436, 33.7378616, -66.6182251, 66.6261978

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0773840, upper bound: 70.0773812
time: 18.05 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0773351, upper bound: 70.0773351
time: 9.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 28.96 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.96
Output dim: 8, lower bound: -70.0722980, upper bound: 70.0722602
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.96
Output dim: 8, lower bound: -70.0722459, upper bound: 70.0722149
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.96
Output dim: 8, lower bound: -70.0773840, upper bound: 70.0773812
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.96
Output dim: 8, lower bound: -70.0773351, upper bound: 70.0773351

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -31.0189953, 24.8025780, -30.9357166, 24.7231255, -55.7421188, 55.7382965
1: -27.5693512, 22.1462631, -27.4605331, 22.0720863, -49.6414337, 49.6067963
2: -35.1581650, 21.8440876, -35.0404625, 21.7984276, -56.9565926, 56.8845520
3: -37.9633064, 18.6281281, -37.8238602, 18.6211052, -56.5844116, 56.4519882
4: -35.6799698, 25.2227497, -35.5156784, 25.1721115, -60.8520775, 60.7384262
5: -30.8432808, 24.0001812, -30.6983452, 23.9106369, -54.7539177, 54.6985245
6: -28.3893318, 28.0441589, -28.3414078, 27.9648819, -56.3542099, 56.3855667
7: -31.3088531, 29.3643150, -31.2222652, 29.2021770, -60.5110321, 60.5865784
8: -43.0630035, 20.8149128, -42.8519974, 20.8916054, -63.9546089, 63.6669006
9: -27.6253929, 27.8147678, -27.5542049, 27.7400475, -55.3654404, 55.3689728

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0703916, upper bound: 70.0703352
time: 8.77 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0701509, upper bound: 70.0700675
time: 10.08 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -31.2972202, 25.0233402, -32.9189682, 26.2862587, -57.5834732, 57.9423065
1: -27.8299522, 22.3449020, -29.2319183, 23.4559517, -51.2859039, 51.5768166
2: -35.4837341, 22.0339336, -37.3041840, 23.1702423, -58.6539764, 59.3381119
3: -38.3056183, 18.7726898, -40.2383041, 19.7631207, -58.0687294, 59.0109940
4: -36.0167046, 25.4451962, -37.7872505, 26.7697563, -62.7864532, 63.2324448
5: -31.1323700, 24.2188263, -32.6631012, 25.4184189, -56.5507889, 56.8819122
6: -28.6357460, 28.2926140, -30.1459332, 29.7435760, -58.3793221, 58.4385414
7: -31.5970688, 29.6405983, -33.2492142, 31.0292892, -62.6263580, 62.8898125
8: -43.4528580, 20.9746151, -45.5151215, 22.2508850, -65.7037201, 66.4897385
9: -27.8751755, 28.0559216, -29.3290710, 29.4981766, -57.3733444, 57.3849945

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0703501, upper bound: 70.0702981
time: 9.94 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0700994, upper bound: 70.0700215
time: 9.24 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -33.6020966, 26.8170280, -33.3717117, 26.6254005, -60.2274971, 60.1887398
1: -29.7752399, 23.9181366, -29.5501919, 23.7519989, -53.5272369, 53.4683266
2: -38.0174789, 23.6465607, -37.7451286, 23.4951611, -61.5126381, 61.3916893
3: -40.9885406, 20.1969490, -40.7016182, 20.0848675, -61.0734100, 60.8985596
4: -38.4543304, 27.3493214, -38.1557350, 27.1784782, -65.6328125, 65.5050430
5: -33.2832565, 25.9151840, -33.0195084, 25.7234993, -59.0067558, 58.9346924
6: -30.7884636, 30.3385963, -30.5937576, 30.1301804, -60.9186325, 60.9323425
7: -33.8945847, 31.5476532, -33.6666222, 31.2852249, -65.1797943, 65.2142715
8: -46.2691574, 22.9063320, -45.9157562, 22.8298798, -69.0990295, 68.8220901
9: -29.9373474, 30.0991535, -29.7352962, 29.8999691, -59.8373070, 59.8344498

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0761924, upper bound: 70.0761988
time: 8.10 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0761158, upper bound: 70.0761145
time: 11.94 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -33.8059120, 26.9792290, -35.4301300, 28.2456532, -62.0515671, 62.4093475
1: -29.9663429, 24.0643082, -31.3720245, 25.1921940, -55.1585388, 55.4363327
2: -38.2589340, 23.7856464, -40.0903625, 24.9199562, -63.1788902, 63.8760071
3: -41.2392387, 20.3009644, -43.2115784, 21.2714577, -62.5106964, 63.5125389
4: -38.7027435, 27.5108356, -40.5017166, 28.8483887, -67.5511169, 68.0125427
5: -33.4988976, 26.0763149, -35.0554962, 27.2852821, -60.7841682, 61.1317978
6: -30.9690609, 30.5205402, -32.4655457, 31.9734268, -62.9424820, 62.9860764
7: -34.1071854, 31.7528706, -35.7676239, 33.1643715, -67.2715530, 67.5204926
8: -46.5557938, 23.0188522, -48.6612930, 24.2641792, -70.8199692, 71.6801453
9: -30.1206627, 30.2755585, -31.5826302, 31.7275047, -61.8481636, 61.8581848

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0761431, upper bound: 70.0761524
time: 10.76 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0760678, upper bound: 70.0760678
time: 9.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.78 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.78
Output dim: 8, lower bound: -70.0703916, upper bound: 70.0703352
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.78
Output dim: 8, lower bound: -70.0701509, upper bound: 70.0700675
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.78
Output dim: 8, lower bound: -70.0703501, upper bound: 70.0702981
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.78
Output dim: 8, lower bound: -70.0700994, upper bound: 70.0700215
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.78
Output dim: 8, lower bound: -70.0761924, upper bound: 70.0761988
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.78
Output dim: 8, lower bound: -70.0761158, upper bound: 70.0761145
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.78
Output dim: 8, lower bound: -70.0761431, upper bound: 70.0761524
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.78
Output dim: 8, lower bound: -70.0760678, upper bound: 70.0760678

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -29.3009262, 23.4620018, -29.6215553, 23.6995811, -53.0005074, 53.0835533
1: -26.0952950, 20.9675827, -26.3353176, 21.1730881, -47.2683792, 47.3029022
2: -33.2625275, 20.6399193, -33.5944328, 20.8791885, -54.1417122, 54.2343521
3: -35.9375458, 17.6186523, -36.2798157, 17.8509712, -53.7885132, 53.8984680
4: -33.8098450, 23.8361855, -34.0901489, 24.1133194, -57.9231606, 57.9263344
5: -29.2006111, 22.7279835, -29.4463081, 22.9379654, -52.1385765, 52.1742935
6: -26.8193512, 26.5188370, -27.1427498, 26.8009510, -53.6203003, 53.6615829
7: -29.6002655, 27.8765583, -29.9170856, 28.0689888, -57.6692543, 57.7936440
8: -40.9198990, 19.5133190, -41.2227783, 19.8912697, -60.8111687, 60.7360916
9: -26.0987339, 26.3062363, -26.3857422, 26.5905628, -52.6892891, 52.6919670

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0672085, upper bound: 70.0670896
time: 10.73 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0670138, upper bound: 70.0669685
time: 11.31 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -32.4258766, 25.9293613, -29.3447762, 23.4828930, -55.9087677, 55.2741318
1: -28.8768349, 23.1566753, -26.0962849, 20.9839497, -49.8607864, 49.2529602
2: -36.8192520, 22.7958736, -33.2859192, 20.6824284, -57.5016785, 56.0817909
3: -39.7498703, 19.4574375, -35.9516830, 17.6912193, -57.4410896, 55.4091187
4: -37.3934975, 26.3810940, -33.7877502, 23.8912086, -61.2847023, 60.1688461
5: -32.2787704, 25.1085358, -29.1820126, 22.7305565, -55.0093231, 54.2905502
6: -29.6657829, 29.3127708, -26.8894386, 26.5576534, -56.2234344, 56.2022095
7: -32.7663612, 30.7735901, -29.6402321, 27.8246765, -60.5910378, 60.4138145
8: -45.1432190, 21.6554527, -40.8738670, 19.6879959, -64.8312073, 62.5293198
9: -28.8936653, 29.0940857, -26.1385765, 26.3493271, -55.2429924, 55.2326546

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0669888, upper bound: 70.0668419
time: 30.41 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0667726, upper bound: 70.0666746
time: 9.13 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -29.5751648, 23.6806335, -31.5543480, 25.2263260, -54.8014832, 55.2349739
1: -26.3537693, 21.1642952, -28.0665569, 22.5223465, -48.8761139, 49.2308502
2: -33.5856514, 20.8273335, -35.8033600, 22.2191887, -55.8048401, 56.6306839
3: -36.2768555, 17.7632980, -38.6292953, 18.9662838, -55.2431412, 56.3925934
4: -34.1434860, 24.0553341, -36.3054581, 25.6681061, -59.8115921, 60.3607941
5: -29.4883213, 22.9438667, -31.3612251, 24.4115810, -53.8999023, 54.3050919
6: -27.0633507, 26.7654381, -28.8991680, 28.5345383, -55.5978851, 55.6646004
7: -29.8853569, 28.1523857, -31.8889389, 29.8551216, -59.7404785, 60.0413246
8: -41.3094711, 19.6667404, -43.8227539, 21.2099152, -62.5193863, 63.4894943
9: -26.3449478, 26.5453568, -28.1106739, 28.3078632, -54.6528091, 54.6560287

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0671804, upper bound: 70.0670700
time: 11.71 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0669802, upper bound: 70.0669456
time: 10.09 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -32.7020340, 26.1496620, -31.2632637, 24.9992714, -57.7013016, 57.4129257
1: -29.1346092, 23.3556767, -27.8171310, 22.3236122, -51.4582214, 51.1728058
2: -37.1415710, 22.9855251, -35.4806442, 22.0146275, -59.1561966, 58.4661713
3: -40.0909538, 19.6028500, -38.2846603, 18.7996101, -58.8905487, 57.8875122
4: -37.7272987, 26.6023254, -35.9897003, 25.4352512, -63.1625519, 62.5920258
5: -32.5661736, 25.3250389, -31.0836582, 24.1943092, -56.7604752, 56.4086914
6: -29.9121513, 29.5605030, -28.6331406, 28.2805576, -58.1927109, 58.1936417
7: -33.0535583, 31.0471458, -31.5996151, 29.6006241, -62.6541824, 62.6467590
8: -45.5299988, 21.8146420, -43.4588203, 20.9965534, -66.5265503, 65.2734604
9: -29.1415920, 29.3349991, -27.8521061, 28.0553284, -57.1969223, 57.1870995

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0669616, upper bound: 70.0668188
time: 10.21 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0667307, upper bound: 70.0666515
time: 10.42 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -31.7125359, 25.3469429, -31.9132271, 25.4958916, -57.2084274, 57.2601700
1: -28.1648388, 22.6203423, -28.3120575, 22.7504539, -50.9152908, 50.9323959
2: -35.9370651, 22.3276329, -36.1449623, 22.4773388, -58.4144058, 58.4725876
3: -38.7595215, 19.0864182, -38.9770126, 19.2307816, -57.9903030, 58.0634308
4: -36.4098358, 25.8152199, -36.5794792, 25.9891930, -62.3990288, 62.3946991
5: -31.4768524, 24.5192509, -31.6280022, 24.6480141, -56.1248589, 56.1472549
6: -29.0607433, 28.6653137, -29.2620697, 28.8419838, -57.9027252, 57.9273796
7: -32.0136681, 29.9256516, -32.2136879, 30.0386467, -62.0523071, 62.1393318
8: -43.9236183, 21.4492455, -44.1114120, 21.7001610, -65.6237640, 65.5606537
9: -28.2519035, 28.4428482, -28.4314270, 28.6245651, -56.8764687, 56.8742752

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0730775, upper bound: 70.0730034
time: 9.52 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0728145, upper bound: 70.0728131
time: 8.57 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -35.1763458, 28.0796776, -31.6641006, 25.3016357, -60.4779816, 59.7437630
1: -31.2372017, 25.0527000, -28.0991116, 22.5793266, -53.8165283, 53.1518097
2: -39.8718224, 24.7181549, -35.8688660, 22.3025055, -62.1743279, 60.5870209
3: -42.9759521, 21.1209984, -38.6804390, 19.0879803, -62.0639267, 59.8014374
4: -40.3653603, 28.6441593, -36.3090439, 25.7900352, -66.1553955, 64.9532013
5: -34.8992844, 27.1503220, -31.3904438, 24.4608536, -59.3601379, 58.5407639
6: -32.2196121, 31.7559700, -29.0358353, 28.6248035, -60.8444138, 60.7917976
7: -35.5137138, 33.1157227, -31.9666519, 29.8202000, -65.3339081, 65.0823746
8: -48.5815735, 23.8506851, -43.7982559, 21.5195694, -70.1011429, 67.6489410
9: -31.3435154, 31.5360508, -28.2089977, 28.4092293, -59.7527428, 59.7450409

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0730131, upper bound: 70.0729295
time: 10.44 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0727476, upper bound: 70.0727306
time: 11.22 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -31.9334564, 25.5216293, -33.9234085, 27.0800095, -59.0134659, 59.4450378
1: -28.3706322, 22.7778244, -30.0972061, 24.1570435, -52.5276756, 52.8750305
2: -36.1958694, 22.4776764, -38.4373512, 23.8708382, -60.0667076, 60.9150276
3: -39.0300827, 19.1998692, -41.4219398, 20.3928566, -59.4229393, 60.6218109
4: -36.6761398, 25.9922390, -38.8719330, 27.6142673, -64.2904053, 64.8641663
5: -31.7094193, 24.6921082, -33.6156120, 26.1766472, -57.8860626, 58.3077164
6: -29.2559013, 28.8617783, -31.0898705, 30.6415920, -59.8974876, 59.9516487
7: -32.2428360, 30.1457176, -34.2623367, 31.8797836, -64.1226120, 64.4080505
8: -44.2308350, 21.5739403, -46.7987595, 23.0952854, -67.3261108, 68.3726883
9: -28.4501762, 28.6329327, -30.2287674, 30.4112701, -58.8614349, 58.8617020

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0730324, upper bound: 70.0729677
time: 9.76 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0727665, upper bound: 70.0727802
time: 9.68 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -35.3673058, 28.2317181, -33.6532860, 26.8698807, -62.2371864, 61.8850021
1: -31.4144135, 25.1900902, -29.8683434, 23.9710045, -55.3854179, 55.0584335
2: -40.0963898, 24.8483200, -38.1403198, 23.6826687, -63.7790451, 62.9886398
3: -43.2108459, 21.2211685, -41.0983238, 20.2397537, -63.4505882, 62.3194923
4: -40.5938606, 28.7975578, -38.5802155, 27.3953056, -67.9891510, 67.3777618
5: -35.0993271, 27.3000088, -33.3588943, 25.9747658, -61.0740929, 60.6589050
6: -32.3887215, 31.9265022, -30.8449001, 30.4074173, -62.7961273, 62.7713928
7: -35.7138977, 33.3049660, -33.9945450, 31.6464310, -67.3603210, 67.2995148
8: -48.8436737, 23.9610519, -46.4627380, 22.8967743, -71.7404480, 70.4237823
9: -31.5164814, 31.7005730, -29.9870605, 30.1785355, -61.6950150, 61.6876335

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0729650, upper bound: 70.0728928
time: 10.60 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0727000, upper bound: 70.0727000
time: 9.44 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.39 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 8, lower bound: -70.0672085, upper bound: 70.0670896
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 8, lower bound: -70.0670138, upper bound: 70.0669685
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 8, lower bound: -70.0669888, upper bound: 70.0668419
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 8, lower bound: -70.0667726, upper bound: 70.0666746
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 8, lower bound: -70.0671804, upper bound: 70.0670700
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 8, lower bound: -70.0669802, upper bound: 70.0669456
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 8, lower bound: -70.0669616, upper bound: 70.0668188
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 8, lower bound: -70.0667307, upper bound: 70.0666515
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 8, lower bound: -70.0730775, upper bound: 70.0730034
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 8, lower bound: -70.0728145, upper bound: 70.0728131
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 8, lower bound: -70.0730131, upper bound: 70.0729295
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 8, lower bound: -70.0727476, upper bound: 70.0727306
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 8, lower bound: -70.0730324, upper bound: 70.0729677
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 8, lower bound: -70.0727665, upper bound: 70.0727802
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 8, lower bound: -70.0729650, upper bound: 70.0728928
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 8, lower bound: -70.0727000, upper bound: 70.0727000

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -28.2588711, 22.6417580, -28.1773376, 22.5649109, -50.8237839, 50.8190918
1: -25.1862946, 20.2399178, -25.0794048, 20.1674023, -45.3536987, 45.3193169
2: -32.1007385, 19.9045906, -31.9881096, 19.8619003, -51.9626350, 51.8927002
3: -34.6854401, 16.9895096, -34.5548019, 16.9803410, -51.6657791, 51.5443115
4: -32.6546555, 22.9894600, -32.4946976, 22.9439297, -55.5985870, 55.4841537
5: -28.1886635, 21.9528103, -28.0521088, 21.8612633, -50.0499191, 50.0049133
6: -25.8698006, 25.5871449, -25.8302593, 25.5120106, -51.3818130, 51.4174004
7: -28.5535946, 26.9599380, -28.4685783, 26.8044338, -55.3580208, 55.4285049
8: -39.5794754, 18.7317734, -39.3754044, 18.8057251, -58.3851929, 58.1071777
9: -25.1745949, 25.3823338, -25.1064358, 25.3097458, -50.4843330, 50.4887695

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0667309, upper bound: 70.0666335
time: 10.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0672088, upper bound: 70.0670896
time: 10.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -28.0389576, 22.4694633, -31.1499271, 24.9190712, -52.9580269, 53.6193924
1: -24.9992676, 20.0832138, -27.7432384, 22.2225552, -47.2218246, 47.8264389
2: -31.8656654, 19.7470474, -35.4063187, 21.9228802, -53.7885437, 55.1533661
3: -34.4200592, 16.8421097, -38.1607742, 18.6865444, -53.1066017, 55.0028839
4: -32.4268494, 22.8057251, -35.9435959, 25.3461952, -57.7730408, 58.7493172
5: -27.9822731, 21.7953224, -30.9958973, 24.1259804, -52.1082535, 52.7912140
6: -25.6689205, 25.3896408, -28.5298119, 28.1900215, -53.8589401, 53.9194527
7: -28.3354397, 26.7857571, -31.5052547, 29.5885696, -57.9240112, 58.2910118
8: -39.3232231, 18.5395012, -43.4437332, 20.8103523, -60.1335754, 61.9832344
9: -24.9792061, 25.1912956, -27.7716026, 27.9933205, -52.9725227, 52.9628983

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0665733, upper bound: 70.0665485
time: 9.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0670138, upper bound: 70.0669685
time: 10.35 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -31.3552685, 25.0873890, -27.9128838, 22.3541718, -53.7094421, 53.0002747
1: -27.9462814, 22.4082775, -24.8487492, 19.9839020, -47.9301834, 47.2570152
2: -35.6280212, 22.0413628, -31.6903877, 19.6709881, -55.2990112, 53.7317467
3: -38.4653473, 18.8132572, -34.2372169, 16.8259144, -55.2912598, 53.0504646
4: -36.2064133, 25.5109768, -32.2012062, 22.7291069, -58.9355202, 57.7121811
5: -31.2432899, 24.3098850, -27.7952919, 21.6619492, -52.9052353, 52.1051712
6: -28.6889744, 28.3574295, -25.5839424, 25.2769375, -53.9659081, 53.9413719
7: -31.6908073, 29.8336945, -28.2015324, 26.5666103, -58.2574120, 58.0352249
8: -43.7691917, 20.8511658, -39.0352097, 18.6113071, -62.3805008, 59.8863754
9: -27.9422989, 28.1453705, -24.8688889, 25.0761547, -53.0184479, 53.0142593

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0665148, upper bound: 70.0663475
time: 10.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0669888, upper bound: 70.0668416
time: 10.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -31.1306000, 24.9112625, -30.8533649, 24.6909561, -55.8215561, 55.7646255
1: -27.7560978, 22.2476864, -27.4904175, 22.0225372, -49.7786331, 49.7381020
2: -35.3882256, 21.8808250, -35.0802917, 21.7160454, -57.1042709, 56.9611092
3: -38.1943245, 18.6636887, -37.8124046, 18.5177097, -56.7120361, 56.4760818
4: -35.9735069, 25.3225307, -35.6241188, 25.1111336, -61.0846405, 60.9466400
5: -31.0334053, 24.1475487, -30.7167511, 23.9056702, -54.9390755, 54.8642998
6: -28.4824371, 28.1563129, -28.2640743, 27.9314423, -56.4138794, 56.4203873
7: -31.4673519, 29.6555099, -31.2120380, 29.3305035, -60.7978516, 60.8675461
8: -43.5072212, 20.6536770, -43.0751648, 20.5973148, -64.1045380, 63.7288437
9: -27.7415237, 27.9500675, -27.5087719, 27.7380791, -55.4795990, 55.4588394

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 198

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0663249, upper bound: 70.0662175
time: 11.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0667726, upper bound: 70.0666745
time: 9.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -28.5437737, 22.8690224, -30.0615158, 24.0557976, -52.5995636, 52.9305382
1: -25.4543114, 20.4442062, -26.7728729, 21.4837437, -46.9380569, 47.2170792
2: -32.4358330, 20.0993042, -34.1457291, 21.1711006, -53.6069336, 54.2450333
3: -35.0381851, 17.1406002, -36.8484344, 18.0707054, -53.1088905, 53.9890327
4: -33.0000038, 23.2175102, -34.6607857, 24.4585342, -57.4585381, 57.8782959
5: -28.4871521, 22.1763458, -29.9232941, 23.2998810, -51.7870331, 52.0996399
6: -26.1233406, 25.8434086, -27.5403538, 27.2067394, -53.3300743, 53.3837547
7: -28.8491230, 27.2456512, -30.3910637, 28.5528297, -57.4019547, 57.6367149
8: -39.9825211, 18.8934574, -41.9229889, 20.0837555, -60.0662727, 60.8164444
9: -25.4310608, 25.6308975, -26.7878075, 26.9897614, -52.4208145, 52.4187012

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0566319, upper bound: 70.0562770
time: 10.05 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0671801, upper bound: 70.0670700
time: 10.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -28.3074875, 22.6838894, -33.0945511, 26.4500771, -54.7575531, 55.7784424
1: -25.2527657, 20.2758942, -29.4761600, 23.5799026, -48.8326645, 49.7520523
2: -32.1822205, 19.9305649, -37.6279068, 23.2685585, -55.4507675, 57.5584717
3: -34.7527313, 16.9838200, -40.5184708, 19.8036003, -54.5563316, 57.5022736
4: -32.7534904, 23.0206718, -38.1688766, 26.9117317, -59.6652107, 61.1895447
5: -28.2645321, 22.0064716, -32.9193726, 25.6054020, -53.8699341, 54.9258423
6: -25.9078827, 25.6311283, -30.2934227, 29.9305439, -55.8384247, 55.9245453
7: -28.6147232, 27.0565681, -33.4824753, 31.3839455, -59.9986649, 60.5390396
8: -39.7044258, 18.6898880, -46.0549240, 22.1385689, -61.8429871, 64.7448120
9: -25.2214470, 25.4256973, -29.5017414, 29.7190056, -54.9404526, 54.9274368

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0564298, upper bound: 70.0561379
time: 10.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0669798, upper bound: 70.0669456
time: 8.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -31.6366062, 25.3117981, -29.7789078, 23.8352356, -55.4718399, 55.0907021
1: -28.2096100, 22.6110916, -26.5291481, 21.2905922, -49.5001984, 49.1402397
2: -35.9563637, 22.2347660, -33.8309288, 20.9709473, -56.9273033, 56.0656891
3: -38.8134995, 18.9618073, -36.5135994, 17.9077606, -56.7212601, 55.4754066
4: -36.5459824, 25.7369251, -34.3525810, 24.2317944, -60.7777786, 60.0895004
5: -31.5362873, 24.5301399, -29.6538277, 23.0882988, -54.6245880, 54.1839676
6: -28.9396515, 28.6103230, -27.2821541, 26.9585419, -55.8981934, 55.8924713
7: -31.9829445, 30.1128960, -30.1088829, 28.3036213, -60.2865601, 60.2217789
8: -44.1635971, 21.0137463, -41.5673370, 19.8765144, -64.0401154, 62.5810776
9: -28.1954212, 28.3904476, -26.5359802, 26.7435112, -54.9389229, 54.9264297

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0588521, upper bound: 70.0587300
time: 11.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0582237, upper bound: 70.0579195
time: 11.99 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -31.3982582, 25.1248665, -32.8037949, 26.2231617, -57.6214104, 57.9286499
1: -28.0073051, 22.4406776, -29.2262077, 23.3814316, -51.3887329, 51.6668739
2: -35.7009811, 22.0648670, -37.3047829, 23.0637474, -58.7647285, 59.3696480
3: -38.5256996, 18.8043137, -40.1737061, 19.6369247, -58.1626205, 58.9780197
4: -36.2973442, 25.5374317, -37.8509064, 26.6792336, -62.9765739, 63.3883362
5: -31.3128033, 24.3574181, -32.6419220, 25.3879547, -56.7007599, 56.9993362
6: -28.7208424, 28.3967476, -30.0275860, 29.6755371, -58.3963699, 58.4243317
7: -31.7457733, 29.9221478, -33.1922226, 31.1285820, -62.8743515, 63.1143608
8: -43.8835030, 20.8066883, -45.6889801, 21.9263630, -65.8098602, 66.4956665
9: -27.9823875, 28.1831989, -29.2430954, 29.4657211, -57.4481087, 57.4262924

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0584647, upper bound: 70.0584575
time: 10.49 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0578842, upper bound: 70.0576805
time: 10.08 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -30.5956955, 24.4706039, -30.3895721, 24.2992477, -54.8949432, 54.8601761
1: -27.1991272, 21.8428116, -26.9939137, 21.6888542, -48.8879814, 48.8367233
2: -34.6970863, 21.5432796, -34.4525032, 21.4066467, -56.1037254, 55.9957771
3: -37.4280205, 18.4141541, -37.1586456, 18.3144741, -55.7424927, 55.5727921
4: -35.1813164, 24.9115257, -34.9045601, 24.7539444, -59.9352608, 59.8160858
5: -30.4011116, 23.6881142, -30.1599159, 23.5131168, -53.9142303, 53.8480186
6: -28.0448647, 27.6720581, -27.8756752, 27.4866791, -55.5315361, 55.5477257
7: -30.8951454, 28.9527664, -30.6876049, 28.7101669, -59.6053123, 59.6403694
8: -42.4989891, 20.6029987, -42.1677475, 20.5443554, -63.0433426, 62.7707443
9: -27.2618790, 27.4546585, -27.0794468, 27.2760258, -54.5379028, 54.5341034

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0724207, upper bound: 70.0723523
time: 9.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0730775, upper bound: 70.0730034
time: 9.13 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -30.3364067, 24.2666321, -33.5040359, 26.7665501, -57.1029358, 57.7706680
1: -26.9788399, 21.6588173, -29.7662582, 23.8493614, -50.8281975, 51.4250717
2: -34.4176140, 21.3579407, -38.0318756, 23.5684319, -57.9860458, 59.3898163
3: -37.1185608, 18.2443199, -40.9298782, 20.1045036, -57.2230644, 59.1741943
4: -34.9110909, 24.6962433, -38.4970589, 27.2811069, -62.1921921, 63.1933022
5: -30.1584721, 23.4985924, -33.2409325, 25.8824539, -56.0409164, 56.7395210
6: -27.8062668, 27.4401951, -30.7041283, 30.2870121, -58.0932770, 58.1443176
7: -30.6358509, 28.7456665, -33.8586464, 31.6150532, -62.2509041, 62.6043129
8: -42.1934013, 20.3783417, -46.4339752, 22.6784630, -64.8718643, 66.8123169
9: -27.0315628, 27.2271843, -29.8675823, 30.0968800, -57.1284409, 57.0947647

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0721708, upper bound: 70.0721817
time: 11.33 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0728145, upper bound: 70.0728131
time: 11.16 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -34.0055580, 27.1645699, -30.1436958, 24.1081734, -58.1137314, 57.3082619
1: -30.2295952, 24.2353611, -26.7838078, 21.5211582, -51.7507553, 51.0191689
2: -38.5747452, 23.8961163, -34.1798172, 21.2335739, -59.8083191, 58.0759277
3: -41.5780716, 20.4189262, -36.8667221, 18.1734200, -59.7514915, 57.2856407
4: -39.0789261, 27.6932983, -34.6367493, 24.5588131, -63.6377411, 62.3300400
5: -33.7707253, 26.2808056, -29.9258060, 23.3288956, -57.0996170, 56.2066116
6: -31.1543083, 30.7165298, -27.6526604, 27.2721615, -58.4264679, 58.3691902
7: -34.3434258, 32.1016235, -30.4433270, 28.4942188, -62.8376427, 62.5449524
8: -47.0975723, 22.9560432, -41.8579254, 20.3667889, -67.4643631, 64.8139648
9: -30.3046646, 30.4990807, -26.8607216, 27.0630970, -57.3677597, 57.3598022

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0667061, upper bound: 70.0666814
time: 11.23 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0663229, upper bound: 70.0662037
time: 11.17 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.75 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0667309, upper bound: 70.0666335
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0672088, upper bound: 70.0670896
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0665733, upper bound: 70.0665485
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0670138, upper bound: 70.0669685
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0665148, upper bound: 70.0663475
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0669888, upper bound: 70.0668416
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0663249, upper bound: 70.0662175
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0667726, upper bound: 70.0666745
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0566319, upper bound: 70.0562770
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0671801, upper bound: 70.0670700
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0564298, upper bound: 70.0561379
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0669798, upper bound: 70.0669456
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0588521, upper bound: 70.0587300
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0582237, upper bound: 70.0579195
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0584647, upper bound: 70.0584575
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0578842, upper bound: 70.0576805
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0724207, upper bound: 70.0723523
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0730775, upper bound: 70.0730034
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0721708, upper bound: 70.0721817
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0728145, upper bound: 70.0728131
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0667061, upper bound: 70.0666814
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.75
Output dim: 8, lower bound: -70.0663229, upper bound: 70.0662037
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.75
Output dim: 8, lower bound: -70.0727476, upper bound: 70.0727306
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.75
Output dim: 8, lower bound: -70.0730324, upper bound: 70.0729677
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.75
Output dim: 8, lower bound: -70.0727665, upper bound: 70.0727802
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.75
Output dim: 8, lower bound: -70.0729650, upper bound: 70.0728928
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.75
Output dim: 8, lower bound: -70.0727000, upper bound: 70.0727000
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=82.72178649902344
rel_dist={8: [-70.08140925617062, 70.08140925618659]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1830.75 seconds
