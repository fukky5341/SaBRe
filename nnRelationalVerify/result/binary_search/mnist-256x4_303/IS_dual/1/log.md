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
execution time: IAR + LP analysis = 1.29 + 8.97 = 10.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -70.0817842, upper bound: 70.0817842


# Binary Search by BASE starts (time budget: 2689.74 seconds, max iter: 100)

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
Binary search time: 42.07 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2647.67 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0588182, upper bound: 70.0607374
time: 8.72 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0528356, upper bound: 70.0528356
time: 5.91 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.78
Output dim: 8, lower bound: -70.0588182, upper bound: 70.0607374
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.78
Output dim: 8, lower bound: -70.0528356, upper bound: 70.0528356

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -38.7710876, 30.8752327, -40.2857857, 32.0787888, -70.8498764, 71.1610184
1: -34.2492867, 27.5032730, -35.5273781, 28.5705376, -62.8198242, 63.0306473
2: -43.8073158, 27.2135506, -45.4768066, 28.2778454, -72.0851517, 72.6903534
3: -47.1584892, 23.1880436, -48.9736557, 24.1043568, -71.2628403, 72.1616974
4: -44.1674614, 31.6087513, -45.8082924, 32.8718872, -77.0393524, 77.4170456
5: -38.3591309, 29.7881165, -39.8284340, 30.9265594, -69.2856903, 69.6165466
6: -35.4926605, 34.8981590, -36.8836212, 36.2383995, -71.7310638, 71.7817841
7: -39.0804977, 36.1019554, -40.5858040, 37.3845253, -76.4650269, 76.6877594
8: -52.9083481, 26.7349491, -54.7539024, 27.9678822, -80.8762283, 81.4888535
9: -34.5444870, 34.6791153, -35.9190674, 36.0372391, -70.5817108, 70.5981827

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0494439, upper bound: 70.0512188
time: 8.72 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0447367, upper bound: 70.0473521
time: 10.51 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -32.2891045, 25.7158127, -38.7683105, 30.8735218, -63.1626282, 64.4841232
1: -28.9310246, 22.9592896, -34.2470627, 27.5002251, -56.4312515, 57.2063370
2: -36.7943077, 22.6683865, -43.8047180, 27.2145729, -64.0088806, 66.4730988
3: -39.4813538, 19.1390572, -47.1526947, 23.1877060, -62.6690598, 66.2917404
4: -37.3771820, 26.2189045, -44.1654205, 31.6098061, -68.9869843, 70.3843231
5: -32.1661224, 24.9640083, -38.3543549, 29.7867451, -61.9528656, 63.3183632
6: -29.5174751, 29.1832962, -35.4915466, 34.8964882, -64.4139557, 64.6748428
7: -32.7392654, 30.8720379, -39.0797195, 36.0985641, -68.8378296, 69.9517593
8: -45.4878120, 21.1242294, -52.9052429, 26.7414131, -72.2292252, 74.0294724
9: -28.6789589, 28.9345284, -34.5456123, 34.6785660, -63.3575134, 63.4801407

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0399477, upper bound: 70.0391276
time: 7.77 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651
time: 14.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.42 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.42
Output dim: 8, lower bound: -70.0494439, upper bound: 70.0512188
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.42
Output dim: 8, lower bound: -70.0447367, upper bound: 70.0473521
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.42
Output dim: 8, lower bound: -70.0399477, upper bound: 70.0391276
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.42
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -37.7340965, 30.0505333, -27.7034836, 22.1330051, -59.8670921, 57.7540169
1: -33.3643417, 26.7689095, -24.6833572, 19.7574348, -53.1217766, 51.4522667
2: -42.6578064, 26.4899521, -31.4493027, 19.5230255, -62.1808319, 57.9392548
3: -45.8947220, 22.5639801, -33.7164955, 16.5821533, -62.4768753, 56.2804718
4: -43.0293770, 30.7501678, -31.8983574, 22.5496254, -65.5790024, 62.6485252
5: -37.3465118, 29.0021133, -27.5186977, 21.4222126, -58.7687225, 56.5208130
6: -34.5389786, 33.9810562, -25.3435059, 25.0820732, -59.6210403, 59.3245621
7: -38.0433235, 35.2016068, -27.9882317, 26.3102531, -64.3535767, 63.1898384
8: -51.6166840, 25.9363155, -38.7757416, 18.5338326, -70.1505127, 64.7120590
9: -33.6054916, 33.7536964, -24.6235199, 24.8312340, -58.4367256, 58.3772163

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0446735, upper bound: 70.0472440
time: 10.45 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0446735, upper bound: 70.0473521
time: 10.05 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -38.0507698, 30.3025627, -32.1802979, 25.6516724, -63.7024422, 62.4828568
1: -33.6366234, 26.9942856, -28.6195812, 22.8838177, -56.5204391, 55.6138687
2: -43.0095291, 26.7100048, -36.4810486, 22.6162281, -65.6257553, 63.1910477
3: -46.2846375, 22.7526741, -39.1781273, 19.2350159, -65.5196457, 61.9307938
4: -43.3790932, 31.0108032, -36.9378738, 26.1555099, -69.5345993, 67.9486694
5: -37.6578178, 29.2445984, -31.9317112, 24.8353806, -62.4931984, 61.1763077
6: -34.8305779, 34.2614174, -29.4439697, 29.0753059, -63.9058838, 63.7053871
7: -38.3610039, 35.4810600, -32.4968910, 30.3835773, -68.7445831, 67.9779510
8: -52.0163765, 26.1715393, -44.6601448, 21.6501999, -73.6665802, 70.8316803
9: -33.8921852, 34.0350075, -28.5943222, 28.8028755, -62.6950569, 62.6293297

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0446735, upper bound: 70.0472440
time: 9.63 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0446735, upper bound: 70.0473521
time: 9.82 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -31.6082611, 25.1799316, -26.7119255, 21.3544388, -52.9626999, 51.8918571
1: -28.3381042, 22.4813957, -23.8354225, 19.0696220, -47.4077263, 46.3168182
2: -36.0331459, 22.1971321, -30.3530502, 18.8328304, -54.8659668, 52.5501823
3: -38.6543808, 18.7242928, -32.5466843, 15.9758701, -54.6302414, 51.2709770
4: -36.6206436, 25.6673851, -30.8234138, 21.7441559, -58.3647995, 56.4907990
5: -31.5013008, 24.4499607, -26.5617599, 20.6817722, -52.1830711, 51.0117188
6: -28.8979645, 28.5738602, -24.4469681, 24.1964436, -53.0944061, 53.0208244
7: -32.0570297, 30.2647324, -27.0057621, 25.4488316, -57.5058594, 57.2704926
8: -44.6175232, 20.6296444, -37.5469208, 17.7655106, -62.3830299, 58.1765671
9: -28.0720577, 28.3322010, -23.7408714, 23.9550056, -52.0270615, 52.0730667

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651
time: 6.29 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651
time: 5.75 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -31.7504768, 25.2933750, -30.9780464, 24.7054405, -56.4559174, 56.2714195
1: -28.4665318, 22.5836143, -27.5906410, 22.0472317, -50.5137634, 50.1742554
2: -36.1961288, 22.2960949, -35.1521492, 21.7772179, -57.9733467, 57.4482422
3: -38.8314629, 18.8102989, -37.7498093, 18.5009193, -57.3323822, 56.5601082
4: -36.7851906, 25.7832642, -35.6360779, 25.1719036, -61.9570923, 61.4193420
5: -31.6439781, 24.5600643, -30.7693272, 23.9351845, -55.5791626, 55.3293877
6: -29.0282612, 28.7034569, -28.3460197, 28.0081329, -57.0363922, 57.0494766
7: -32.2036209, 30.3994598, -31.3037338, 29.3507957, -61.5544052, 61.7031937
8: -44.8099709, 20.7254333, -43.1750946, 20.7050724, -65.5150375, 63.9005280
9: -28.2007637, 28.4605713, -27.5249062, 27.7344646, -55.9352264, 55.9854774

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651
time: 6.52 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651
time: 18.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.24 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.24
Output dim: 8, lower bound: -70.0446735, upper bound: 70.0472440
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.24
Output dim: 8, lower bound: -70.0446735, upper bound: 70.0473521
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.24
Output dim: 8, lower bound: -70.0446735, upper bound: 70.0472440
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.24
Output dim: 8, lower bound: -70.0446735, upper bound: 70.0473521
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.24
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.24
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.24
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.24
Output dim: 8, lower bound: -70.0369651, upper bound: 70.0369651

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -26.6202583, 21.2837868, -27.7034836, 22.1330051, -48.7532616, 48.9872704
1: -23.7588882, 19.0071583, -24.6833572, 19.7574348, -43.5163193, 43.6905136
2: -30.2525902, 18.7684727, -31.4493027, 19.5230255, -49.7756157, 50.2177734
3: -32.4382744, 15.9198036, -33.7164955, 16.5821533, -49.0204277, 49.6362991
4: -30.7266541, 21.6690903, -31.8983574, 22.5496254, -53.2762794, 53.5674477
5: -26.4748993, 20.6141663, -27.5186977, 21.4222126, -47.8971100, 48.1328659
6: -24.3619461, 24.1155949, -25.3435059, 25.0820732, -49.4440155, 49.4590988
7: -26.9148216, 25.3717041, -27.9882317, 26.3102531, -53.2250748, 53.3599358
8: -37.4352570, 17.6913090, -38.7757416, 18.5338326, -55.9690857, 56.4670486
9: -23.6589355, 23.8746700, -24.6235199, 24.8312340, -48.4901657, 48.4981918

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0473515, upper bound: 70.0488490
time: 9.08 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0462847, upper bound: 70.0480304
time: 10.89 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -31.0836563, 24.7902832, -27.7034836, 22.1330051, -53.2166557, 52.4937668
1: -27.6796188, 22.1215172, -24.6833572, 19.7574348, -47.4370461, 46.8048744
2: -35.2675552, 21.8497868, -31.4493027, 19.5230255, -54.7905807, 53.2990837
3: -37.8789978, 18.5659542, -33.7164955, 16.5821533, -54.4611473, 52.2824478
4: -35.7516823, 25.2560539, -31.8983574, 22.5496254, -58.3013077, 57.1544113
5: -30.8740349, 24.0161915, -27.5186977, 21.4222126, -52.2962341, 51.5348892
6: -28.4426403, 28.1006145, -25.3435059, 25.0820732, -53.5247116, 53.4441223
7: -31.4064445, 29.4421825, -27.9882317, 26.3102531, -57.7166977, 57.4304085
8: -43.3041649, 20.7807980, -38.7757416, 18.5338326, -61.8379898, 59.5565338
9: -27.6168251, 27.8261929, -24.6235199, 24.8312340, -52.4480591, 52.4497147

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0471766, upper bound: 70.0491074
time: 11.24 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0462847, upper bound: 70.0480304
time: 11.82 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -26.6202583, 21.2837868, -32.1802979, 25.6516724, -52.2719307, 53.4640846
1: -23.7588882, 19.0071583, -28.6195812, 22.8838177, -46.6427078, 47.6267395
2: -30.2525902, 18.7684727, -36.4810486, 22.6162281, -52.8688126, 55.2495193
3: -32.4382744, 15.9198036, -39.1781273, 19.2350159, -51.6732864, 55.0979309
4: -30.7266541, 21.6690903, -36.9378738, 26.1555099, -56.8821640, 58.6069603
5: -26.4748993, 20.6141663, -31.9317112, 24.8353806, -51.3102798, 52.5458755
6: -24.3619461, 24.1155949, -29.4439697, 29.0753059, -53.4372520, 53.5595627
7: -26.9148216, 25.3717041, -32.4968910, 30.3835773, -57.2983932, 57.8685913
8: -37.4352570, 17.6913090, -44.6601448, 21.6501999, -59.0854568, 62.3514557
9: -23.6589355, 23.8746700, -28.5943222, 28.8028755, -52.4618034, 52.4689903

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0426424, upper bound: 70.0449248
time: 6.73 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0405922, upper bound: 70.0432108
time: 12.21 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -31.0836563, 24.7902832, -32.1802979, 25.6516724, -56.7353287, 56.9705734
1: -27.6796188, 22.1215172, -28.6195812, 22.8838177, -50.5634384, 50.7410965
2: -35.2675552, 21.8497868, -36.4810486, 22.6162281, -57.8837814, 58.3308334
3: -37.8789978, 18.5659542, -39.1781273, 19.2350159, -57.1139984, 57.7440758
4: -35.7516823, 25.2560539, -36.9378738, 26.1555099, -61.9071922, 62.1939278
5: -30.8740349, 24.0161915, -31.9317112, 24.8353806, -55.7094154, 55.9479027
6: -28.4426403, 28.1006145, -29.4439697, 29.0753059, -57.5179443, 57.5445862
7: -31.4064445, 29.4421825, -32.4968910, 30.3835773, -61.7900238, 61.9390717
8: -43.3041649, 20.7807980, -44.6601448, 21.6501999, -64.9543610, 65.4409409
9: -27.6168251, 27.8261929, -28.5943222, 28.8028755, -56.4196968, 56.4205132

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0426424, upper bound: 70.0449975
time: 8.88 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0405922, upper bound: 70.0433239
time: 9.86 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -23.8933449, 19.1247635, -26.7119255, 21.3544388, -45.2477837, 45.8366890
1: -21.5397129, 17.0840530, -23.8354225, 19.0696220, -40.6093369, 40.9194756
2: -27.3195496, 16.8319569, -30.3530502, 18.8328304, -46.1523819, 47.1850052
3: -29.2478065, 14.1104231, -32.5466843, 15.9758701, -45.2236748, 46.6571083
4: -27.8731976, 19.4391842, -30.8234138, 21.7441559, -49.6173515, 50.2625961
5: -23.9226494, 18.6305428, -26.5617599, 20.6817722, -44.6044235, 45.1923027
6: -21.8609161, 21.6355057, -24.4469681, 24.1964436, -46.0573578, 46.0824738
7: -24.2307262, 23.2052383, -27.0057621, 25.4488316, -49.6795502, 50.2109985
8: -34.4005241, 15.2866468, -37.5469208, 17.7655106, -52.1660309, 52.8335648
9: -21.2250462, 21.4622612, -23.7408714, 23.9550056, -45.1800537, 45.2031250

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0372447, upper bound: 70.0361963
time: 9.87 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0364158, upper bound: 70.0355220
time: 9.17 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -26.5051231, 21.2003975, -26.7119255, 21.3544388, -47.8595581, 47.9123230
1: -23.9098358, 18.9253578, -23.8354225, 19.0696220, -42.9794579, 42.7607803
2: -30.3303452, 18.6445999, -30.3530502, 18.8328304, -49.1631737, 48.9976501
3: -32.4786453, 15.6348019, -32.5466843, 15.9758701, -48.4545135, 48.1814880
4: -30.9363499, 21.5362873, -30.8234138, 21.7441559, -52.6805038, 52.3597031
5: -26.5438118, 20.6512108, -26.5617599, 20.6817722, -47.2255859, 47.2129707
6: -24.2591343, 24.0123444, -24.4469681, 24.1964436, -48.4555779, 48.4593124
7: -26.9261360, 25.7189884, -27.0057621, 25.4488316, -52.3749619, 52.7247505
8: -38.0609360, 16.9489403, -37.5469208, 17.7655106, -55.8264465, 54.4958611
9: -23.5536003, 23.8045216, -23.7408714, 23.9550056, -47.5086060, 47.5453796

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0372447, upper bound: 70.0361963
time: 9.03 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0364158, upper bound: 70.0355220
time: 8.87 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -23.8933449, 19.1247635, -30.9780464, 24.7054405, -48.5987854, 50.1028099
1: -21.5397129, 17.0840530, -27.5906410, 22.0472317, -43.5869446, 44.6746902
2: -27.3195496, 16.8319569, -35.1521492, 21.7772179, -49.0967636, 51.9841042
3: -29.2478065, 14.1104231, -37.7498093, 18.5009193, -47.7487259, 51.8602333
4: -27.8731976, 19.4391842, -35.6360779, 25.1719036, -53.0451012, 55.0752640
5: -23.9226494, 18.6305428, -30.7693272, 23.9351845, -47.8578262, 49.3998718
6: -21.8609161, 21.6355057, -28.3460197, 28.0081329, -49.8690491, 49.9815254
7: -24.2307262, 23.2052383, -31.3037338, 29.3507957, -53.5815125, 54.5089722
8: -34.4005241, 15.2866468, -43.1750946, 20.7050724, -55.1055870, 58.4617271
9: -21.2250462, 21.4622612, -27.5249062, 27.7344646, -48.9595108, 48.9871674

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0303963, upper bound: 70.0296313
time: 7.75 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0278375, upper bound: 70.0278375
time: 6.87 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -26.5051231, 21.2003975, -30.9780464, 24.7054405, -51.2105637, 52.1784439
1: -23.9098358, 18.9253578, -27.5906410, 22.0472317, -45.9570694, 46.5159988
2: -30.3303452, 18.6445999, -35.1521492, 21.7772179, -52.1075592, 53.7967453
3: -32.4786453, 15.6348019, -37.7498093, 18.5009193, -50.9795647, 53.3846092
4: -30.9363499, 21.5362873, -35.6360779, 25.1719036, -56.1082458, 57.1723633
5: -26.5438118, 20.6512108, -30.7693272, 23.9351845, -50.4789810, 51.4205399
6: -24.2591343, 24.0123444, -28.3460197, 28.0081329, -52.2672653, 52.3583641
7: -26.9261360, 25.7189884, -31.3037338, 29.3507957, -56.2769241, 57.0227127
8: -38.0609360, 16.9489403, -43.1750946, 20.7050724, -58.7660065, 60.1240273
9: -23.5536003, 23.8045216, -27.5249062, 27.7344646, -51.2880630, 51.3294220

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0343853, upper bound: 70.0340388
time: 8.62 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
time: 8.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.28 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -70.0473515, upper bound: 70.0488490
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -70.0462847, upper bound: 70.0480304
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -70.0471766, upper bound: 70.0491074
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -70.0462847, upper bound: 70.0480304
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -70.0426424, upper bound: 70.0449248
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -70.0405922, upper bound: 70.0432108
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -70.0426424, upper bound: 70.0449975
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -70.0405922, upper bound: 70.0433239
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -70.0372447, upper bound: 70.0361963
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -70.0364158, upper bound: 70.0355220
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -70.0372447, upper bound: 70.0361963
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -70.0364158, upper bound: 70.0355220
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -70.0303963, upper bound: 70.0296313
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -70.0278375, upper bound: 70.0278375
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -70.0343853, upper bound: 70.0340388
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.28
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -26.6202583, 21.2837868, -26.3270664, 21.0619926, -47.6822433, 47.6108551
1: -23.7588882, 19.0071583, -23.4863262, 18.8001976, -42.5590820, 42.4934769
2: -30.2525902, 18.7684727, -29.9234428, 18.5574989, -48.8100815, 48.6919174
3: -32.4382744, 15.9198036, -32.0835609, 15.7510319, -48.1893082, 48.0033646
4: -30.7266541, 21.6690903, -30.3834648, 21.4353180, -52.1619720, 52.0525436
5: -26.4748993, 20.6141663, -26.1863155, 20.4032288, -46.8781281, 46.8004799
6: -24.3619461, 24.1155949, -24.1003876, 23.8509064, -48.2128525, 48.2159805
7: -26.9148216, 25.3717041, -26.6125298, 25.1017456, -52.0165558, 51.9842300
8: -37.4352570, 17.6913090, -37.0155945, 17.5048084, -54.9400597, 54.7069016
9: -23.6589355, 23.8746700, -23.4125290, 23.6149788, -47.2739105, 47.2871971

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0484481, upper bound: 70.0503841
time: 9.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0484481, upper bound: 70.0503841
time: 10.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -26.4200439, 21.1281090, -29.4991741, 23.5584431, -49.9784775, 50.6272812
1: -23.5848522, 18.8672619, -26.3270702, 20.9946251, -44.5794754, 45.1943283
2: -30.0325432, 18.6278381, -33.5733871, 20.7533035, -50.7858467, 52.2012215
3: -32.2021103, 15.7979412, -35.9496231, 17.5792770, -49.7813797, 51.7475662
4: -30.5079365, 21.5066757, -34.0572624, 23.9947891, -54.5027199, 55.5639381
5: -26.2823677, 20.4667072, -29.3300667, 22.8123169, -49.0946846, 49.7967720
6: -24.1814041, 23.9358940, -26.9707870, 26.7130032, -50.8943977, 50.9066734
7: -26.7153168, 25.1985130, -29.8611832, 28.0794983, -54.7948151, 55.0596886
8: -37.1819839, 17.5391293, -41.3526497, 19.6187954, -56.8007774, 58.8917770
9: -23.4833202, 23.6985664, -26.2421398, 26.4658318, -49.9491501, 49.9407043

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0484481, upper bound: 70.0503841
time: 8.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0484481, upper bound: 70.0503841
time: 8.45 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -29.6318283, 23.6587715, -27.7034836, 22.1330051, -51.7648315, 51.3622551
1: -26.4122143, 21.1132393, -24.6833572, 19.7574348, -46.1696472, 45.7965965
2: -33.6573181, 20.8311291, -31.4493027, 19.5230255, -53.1803436, 52.2804337
3: -36.1476898, 17.6926975, -33.7164955, 16.5821533, -52.7298431, 51.4091949
4: -34.1554222, 24.0789452, -31.8983574, 22.5496254, -56.7050476, 55.9773026
5: -29.4699936, 22.9421234, -27.5186977, 21.4222126, -50.8921890, 50.4608154
6: -27.1273632, 26.8055820, -25.3435059, 25.0820732, -52.2094345, 52.1490860
7: -29.9553490, 28.1725216, -27.9882317, 26.3102531, -56.2656021, 56.1607513
8: -41.4522133, 19.6985435, -38.7757416, 18.5338326, -59.9860458, 58.4742813
9: -26.3399734, 26.5415039, -24.6235199, 24.8312340, -51.1712074, 51.1650238

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0462847, upper bound: 70.0480304
time: 9.34 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0462847, upper bound: 70.0480304
time: 11.04 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -32.6768112, 26.0552807, -27.4863739, 21.9645424, -54.6413536, 53.5416565
1: -29.1556416, 23.2251358, -24.4959679, 19.6063938, -48.7620354, 47.7210999
2: -37.1657448, 22.9396610, -31.2102032, 19.3700066, -56.5357513, 54.1498566
3: -39.8599319, 19.4477100, -33.4585876, 16.4485855, -56.3085175, 52.9062958
4: -37.6865311, 26.5414772, -31.6626148, 22.3726654, -60.0591965, 58.2040939
5: -32.4948311, 25.2508488, -27.3093739, 21.2623634, -53.7571945, 52.5602112
6: -29.8882027, 29.5550022, -25.1460762, 24.8885174, -54.7767181, 54.7010727
7: -33.0743217, 31.0301323, -27.7716560, 26.1237030, -59.1980247, 58.8017883
8: -45.6161690, 21.7323742, -38.5036278, 18.3653164, -63.9814835, 60.2360001
9: -29.0565319, 29.2864475, -24.4327488, 24.6397400, -53.6962662, 53.7191963

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0462847, upper bound: 70.0480304
time: 11.18 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0462847, upper bound: 70.0480304
time: 9.99 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -26.6202583, 21.2837868, -30.6713352, 24.4740524, -51.0942993, 51.9551239
1: -23.7588882, 19.0071583, -27.3061256, 21.8331375, -45.5920219, 46.3132782
2: -30.2525902, 18.7684727, -34.8056984, 21.5560188, -51.8086090, 53.5741730
3: -32.4382744, 15.9198036, -37.3802109, 18.3251591, -50.7634239, 53.3000145
4: -30.7266541, 21.6690903, -35.2799683, 24.9274406, -55.6540909, 56.9490585
5: -26.4748993, 20.6141663, -30.4711876, 23.7177200, -50.1926193, 51.0853539
6: -24.3619461, 24.1155949, -28.0755959, 27.7323093, -52.0942459, 52.1911888
7: -26.9148216, 25.3717041, -30.9872379, 29.0692291, -55.9840317, 56.3589401
8: -37.4352570, 17.6913090, -42.7393875, 20.5060844, -57.9413338, 60.4306870
9: -23.6589355, 23.8746700, -27.2662354, 27.4623585, -51.1212921, 51.1409035

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0410477, upper bound: 70.0436042
time: 12.18 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0410477, upper bound: 70.0436042
time: 9.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -26.4200439, 21.1281090, -33.8331490, 26.9714317, -53.3914719, 54.9612579
1: -23.5848522, 18.8672619, -30.1337051, 24.0348454, -47.6196976, 49.0009575
2: -30.0325432, 18.6278381, -38.4438782, 23.7475338, -53.7800751, 57.0717125
3: -32.2021103, 15.7979412, -41.2264977, 20.1497726, -52.3518829, 57.0244370
4: -30.5079365, 21.5066757, -38.9315491, 27.4957790, -58.0037117, 60.4382248
5: -26.2823677, 20.4667072, -33.6039658, 26.1256065, -52.4079742, 54.0706673
6: -24.1814041, 23.9358940, -30.9441319, 30.5789375, -54.7603378, 54.8800240
7: -26.7153168, 25.1985130, -34.2123604, 32.0176010, -58.7329178, 59.4108734
8: -37.1819839, 17.5391293, -47.0670624, 22.6634483, -59.8454208, 64.6061935
9: -23.4833202, 23.6985664, -30.0943279, 30.3326874, -53.8160019, 53.7928886

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0410477, upper bound: 70.0436042
time: 11.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0410477, upper bound: 70.0436042
time: 8.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -31.0836563, 24.7902832, -30.6713352, 24.4740524, -55.5576935, 55.4616165
1: -27.6796188, 22.1215172, -27.3061256, 21.8331375, -49.5127525, 49.4276390
2: -35.2675552, 21.8497868, -34.8056984, 21.5560188, -56.8235741, 56.6554871
3: -37.8789978, 18.5659542, -37.3802109, 18.3251591, -56.2041435, 55.9461670
4: -35.7516823, 25.2560539, -35.2799683, 24.9274406, -60.6791229, 60.5360222
5: -30.8740349, 24.0161915, -30.4711876, 23.7177200, -54.5917511, 54.4873810
6: -28.4426403, 28.1006145, -28.0755959, 27.7323093, -56.1749496, 56.1762085
7: -31.4064445, 29.4421825, -30.9872379, 29.0692291, -60.4756699, 60.4294128
8: -43.3041649, 20.7807980, -42.7393875, 20.5060844, -63.8102417, 63.5201721
9: -27.6168251, 27.8261929, -27.2662354, 27.4623585, -55.0791779, 55.0924263

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0406572, upper bound: 70.0433239
time: 9.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0406572, upper bound: 70.0433239
time: 10.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -30.8547783, 24.6116695, -33.8331490, 26.9714317, -57.8262100, 58.4448166
1: -27.4805851, 21.9620972, -30.1337051, 24.0348454, -51.5154305, 52.0957985
2: -35.0152054, 21.6881771, -38.4438782, 23.7475338, -58.7627411, 60.1320534
3: -37.6064453, 18.4264069, -41.2264977, 20.1497726, -57.7562180, 59.6529045
4: -35.5027008, 25.0690460, -38.9315491, 27.4957790, -62.9984779, 64.0005951
5: -30.6537876, 23.8475342, -33.6039658, 26.1256065, -56.7793961, 57.4514923
6: -28.2348766, 27.8966064, -30.9441319, 30.5789375, -58.8138123, 58.8407364
7: -31.1777000, 29.2453766, -34.2123604, 32.0176010, -63.1952972, 63.4577370
8: -43.0169373, 20.6043453, -47.0670624, 22.6634483, -65.6803741, 67.6714096
9: -27.4153748, 27.6238289, -30.0943279, 30.3326874, -57.7480545, 57.7181549

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0406572, upper bound: 70.0433239
time: 7.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0406572, upper bound: 70.0433239
time: 9.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -23.8933449, 19.1247635, -25.3963356, 20.3319473, -44.2252922, 44.5210953
1: -21.5397129, 17.0840530, -22.6886101, 18.1544571, -39.6941681, 39.7726555
2: -27.3195496, 16.8319569, -28.8947372, 17.9141331, -45.2336807, 45.7266846
3: -29.2478065, 14.1104231, -30.9884109, 15.1855583, -44.4333649, 45.0988350
4: -27.8731976, 19.4391842, -29.3716030, 20.6856575, -48.5588531, 48.8107872
5: -23.9226494, 18.6305428, -25.2920303, 19.7078781, -43.6305199, 43.9225731
6: -21.8609161, 21.6355057, -23.2637863, 23.0180759, -44.8789902, 44.8992844
7: -24.2307262, 23.2052383, -25.6896286, 24.2927933, -48.5235138, 48.8948631
8: -34.4005241, 15.2866468, -35.8561516, 16.7959003, -51.1964111, 51.1427994
9: -21.2250462, 21.4622612, -22.5867672, 22.7974529, -44.0224991, 44.0490265

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 126

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0385612, upper bound: 70.0379525
time: 6.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0368249, upper bound: 70.0366542
time: 6.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -23.7296371, 18.9978638, -28.5106983, 22.7852592, -46.5148888, 47.5085602
1: -21.3965168, 16.9696808, -25.4804077, 20.3094292, -41.7059402, 42.4500885
2: -27.1387062, 16.7174702, -32.4774704, 20.0638332, -47.2025375, 49.1949387
3: -29.0516281, 14.0104027, -34.7816315, 16.9764366, -46.0280647, 48.7920303
4: -27.6904068, 19.3073215, -32.9778824, 23.1865959, -50.8769989, 52.2851944
5: -23.7646618, 18.5096302, -28.3745270, 22.0662155, -45.8308716, 46.8841553
6: -21.7133789, 21.4885216, -26.0703659, 25.8269596, -47.5403214, 47.5588875
7: -24.0661221, 23.0596657, -28.8762627, 27.2141685, -51.2802887, 51.9359283
8: -34.1859779, 15.1697884, -40.1135941, 18.8553925, -53.0413704, 55.2833824
9: -21.0811977, 21.3189468, -25.3526134, 25.5897045, -46.6708984, 46.6715546

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0375593, upper bound: 70.0372489
time: 9.47 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0360782, upper bound: 70.0360782
time: 8.24 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -26.5051231, 21.2003975, -25.3963356, 20.3319473, -46.8370705, 46.5967331
1: -23.9098358, 18.9253578, -22.6886101, 18.1544571, -42.0642929, 41.6139641
2: -30.3303452, 18.6445999, -28.8947372, 17.9141331, -48.2444763, 47.5393295
3: -32.4786453, 15.6348019, -30.9884109, 15.1855583, -47.6642036, 46.6232109
4: -30.9363499, 21.5362873, -29.3716030, 20.6856575, -51.6220016, 50.9078903
5: -26.5438118, 20.6512108, -25.2920303, 19.7078781, -46.2516899, 45.9432411
6: -24.2591343, 24.0123444, -23.2637863, 23.0180759, -47.2772102, 47.2761192
7: -26.9261360, 25.7189884, -25.6896286, 24.2927933, -51.2189255, 51.4086113
8: -38.0609360, 16.9489403, -35.8561516, 16.7959003, -54.8568344, 52.8050919
9: -23.5536003, 23.8045216, -22.5867672, 22.7974529, -46.3510513, 46.3912849

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0364158, upper bound: 70.0355220
time: 7.20 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0364158, upper bound: 70.0355220
time: 8.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -26.3311081, 21.0652733, -28.5106983, 22.7852592, -49.1163635, 49.5759735
1: -23.7575264, 18.8037262, -25.4804077, 20.3094292, -44.0669556, 44.2841339
2: -30.1383629, 18.5223179, -32.4774704, 20.0638332, -50.2021942, 50.9997864
3: -32.2695923, 15.5290442, -34.7816315, 16.9764366, -49.2460289, 50.3106766
4: -30.7417469, 21.3959961, -32.9778824, 23.1865959, -53.9283333, 54.3738785
5: -26.3758297, 20.5225220, -28.3745270, 22.0662155, -48.4420471, 48.8970490
6: -24.1024933, 23.8557415, -26.0703659, 25.8269596, -49.9294434, 49.9261093
7: -26.7503891, 25.5645905, -28.8762627, 27.2141685, -53.9645576, 54.4408531
8: -37.8334122, 16.8244743, -40.1135941, 18.8553925, -56.6888046, 56.9380646
9: -23.4006271, 23.6508446, -25.3526134, 25.5897045, -48.9903259, 49.0034561

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of IS_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0301603, upper bound: 70.0287249
time: 10.28 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0287074, upper bound: 70.0275310
time: 9.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -23.8933449, 19.1247635, -29.7929344, 23.7936401, -47.6869850, 48.9176903
1: -21.5397129, 17.0840530, -26.5533142, 21.2175999, -42.7573128, 43.6373520
2: -27.3195496, 16.8319569, -33.8510666, 20.9506454, -48.2701950, 50.6830215
3: -29.2478065, 14.1104231, -36.3417740, 17.7769928, -47.0247993, 50.4521942
4: -27.8731976, 19.4391842, -34.3541718, 24.2056293, -52.0788269, 53.7933578
5: -23.9226494, 18.6305428, -29.6358414, 23.0600128, -46.9826584, 48.2663803
6: -21.8609161, 21.6355057, -27.2640285, 26.9492607, -48.8101768, 48.8995361
7: -24.2307262, 23.2052383, -30.1243706, 28.3306599, -52.5613785, 53.3296089
8: -34.4005241, 15.2866468, -41.6844406, 19.7884827, -54.1889954, 56.9710770
9: -21.2250462, 21.4622612, -26.4818764, 26.6885452, -47.9135895, 47.9441376

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0322538, upper bound: 70.0323689
time: 7.03 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0302314, upper bound: 70.0308055
time: 8.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -23.5907135, 18.8917866, -33.9506302, 27.0796318, -50.6703415, 52.8424149
1: -21.2737236, 16.8700619, -30.1606674, 24.0396118, -45.3133316, 47.0307312
2: -26.9843292, 16.6196842, -38.5107040, 23.8788548, -50.8631821, 55.1303749
3: -28.8875885, 13.9240665, -41.3327408, 20.2305222, -49.1181107, 55.2568054
4: -27.5385246, 19.1951408, -38.9733200, 27.5294991, -55.0680237, 58.1684608
5: -23.6326962, 18.4062099, -33.7068405, 26.1671982, -49.7998810, 52.1130409
6: -21.5857544, 21.3630619, -31.0076351, 30.6396561, -52.2254105, 52.3706856
7: -23.9273033, 22.9362373, -34.2150955, 32.0846062, -56.0119057, 57.1513329
8: -34.0064545, 15.0626831, -47.1107750, 22.7384033, -56.7448425, 62.1734581
9: -20.9581871, 21.1971245, -30.1267128, 30.3925209, -51.3507080, 51.3238373

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 126

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0289307, upper bound: 70.0297877
time: 7.34 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0275310, upper bound: 70.0287074
time: 7.10 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 15.79 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0484481, upper bound: 70.0503841
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0484481, upper bound: 70.0503841
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0484481, upper bound: 70.0503841
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0484481, upper bound: 70.0503841
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0462847, upper bound: 70.0480304
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0462847, upper bound: 70.0480304
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0462847, upper bound: 70.0480304
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0462847, upper bound: 70.0480304
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0410477, upper bound: 70.0436042
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0410477, upper bound: 70.0436042
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0410477, upper bound: 70.0436042
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0410477, upper bound: 70.0436042
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0406572, upper bound: 70.0433239
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0406572, upper bound: 70.0433239
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0406572, upper bound: 70.0433239
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0406572, upper bound: 70.0433239
IS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0385612, upper bound: 70.0379525
IS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0368249, upper bound: 70.0366542
IS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0375593, upper bound: 70.0372489
IS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0360782, upper bound: 70.0360782
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0364158, upper bound: 70.0355220
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0364158, upper bound: 70.0355220
IS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0301603, upper bound: 70.0287249
IS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0287074, upper bound: 70.0275310
IS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0322538, upper bound: 70.0323689
IS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0302314, upper bound: 70.0308055
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0289307, upper bound: 70.0297877
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 15.79
Output dim: 8, lower bound: -70.0275310, upper bound: 70.0287074
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.79
Output dim: 8, lower bound: -70.0343853, upper bound: 70.0340388
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.79
Output dim: 8, lower bound: -70.0328716, upper bound: 70.0328716
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=82.72178649902344
rel_dist={8: [-70.0816870801838, 70.08168706258579]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0558967, upper bound: 70.0569274
time: 9.88 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0528046, upper bound: 70.0528046
time: 8.16 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.19 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 18.19
Output dim: 8, lower bound: -70.0558967, upper bound: 70.0569274
IS_A2, status: Status.UNKNOWN, split count: 1, time: 18.19
Output dim: 8, lower bound: -70.0528046, upper bound: 70.0528046

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -38.7710876, 30.8752327, -40.2857857, 32.0787888, -70.8498764, 71.1610184
1: -34.2492867, 27.5032730, -35.5273781, 28.5705376, -62.8198242, 63.0306473
2: -43.8073158, 27.2135506, -45.4768066, 28.2778454, -72.0851517, 72.6903534
3: -47.1584892, 23.1880436, -48.9736557, 24.1043568, -71.2628403, 72.1616974
4: -44.1674614, 31.6087513, -45.8082924, 32.8718872, -77.0393524, 77.4170456
5: -38.3591309, 29.7881165, -39.8284340, 30.9265594, -69.2856903, 69.6165466
6: -35.4926605, 34.8981590, -36.8836212, 36.2383995, -71.7310638, 71.7817841
7: -39.0804977, 36.1019554, -40.5858040, 37.3845253, -76.4650269, 76.6877594
8: -52.9083481, 26.7349491, -54.7539024, 27.9678822, -80.8762283, 81.4888535
9: -34.5444870, 34.6791153, -35.9190674, 36.0372391, -70.5817108, 70.5981827

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0454561, upper bound: 70.0463553
time: 11.07 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0412921, upper bound: 70.0428926
time: 11.75 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -32.2891045, 25.7158127, -36.6531487, 29.1904869, -61.4795876, 62.3689613
1: -28.9310246, 22.9592896, -32.4586220, 26.0101051, -54.9411316, 55.4179115
2: -36.7943077, 22.6683865, -41.4753036, 25.7377625, -62.5320625, 64.1436768
3: -39.4813538, 19.1390572, -44.6119843, 21.9138203, -61.3951721, 63.7510376
4: -37.3771820, 26.2189045, -41.8728943, 29.8531857, -67.2303619, 68.0917969
5: -32.1661224, 24.9640083, -36.3005295, 28.1997452, -60.3658638, 61.2645378
6: -29.5174751, 29.1832962, -33.5507278, 33.0291939, -62.5466576, 62.7340202
7: -32.7392654, 30.8720379, -36.9767609, 34.3027039, -67.0419693, 67.8488007
8: -45.4878120, 21.1242294, -50.3245544, 25.0454254, -70.5332336, 71.4487762
9: -28.6789589, 28.9345284, -32.6344261, 32.7912445, -61.4701920, 61.5689545

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 80

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0387142, upper bound: 70.0381409
time: 9.76 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369573, upper bound: 70.0369573
time: 8.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.67 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 19.67
Output dim: 8, lower bound: -70.0454561, upper bound: 70.0463553
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 19.67
Output dim: 8, lower bound: -70.0412921, upper bound: 70.0428926
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 19.67
Output dim: 8, lower bound: -70.0387142, upper bound: 70.0381409
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 19.67
Output dim: 8, lower bound: -70.0369573, upper bound: 70.0369573

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -34.2098160, 27.2508984, -27.7034836, 22.1330051, -56.3428154, 54.9543800
1: -30.3521690, 24.2862873, -24.6833572, 19.7574348, -50.1096039, 48.9696426
2: -38.7434540, 24.0322189, -31.4493027, 19.5230255, -58.2664795, 55.4815216
3: -41.6086617, 20.4520206, -33.7164955, 16.5821533, -58.1908150, 54.1685181
4: -39.1557961, 27.8362617, -31.8983574, 22.5496254, -61.7054214, 59.7346191
5: -33.8975029, 26.3424797, -27.5186977, 21.4222126, -55.3197098, 53.8611679
6: -31.2985802, 30.8679581, -25.3435059, 25.0820732, -56.3806534, 56.2114563
7: -34.5168800, 32.1320610, -27.9882317, 26.3102531, -60.8271332, 60.1202927
8: -47.2089233, 23.2356701, -38.7757416, 18.5338326, -65.7427521, 62.0114136
9: -30.4190598, 30.6145172, -24.6235199, 24.8312340, -55.2502937, 55.2380371

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 80

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0412469, upper bound: 70.0428406
time: 10.32 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0412469, upper bound: 70.0428926
time: 11.68 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -35.2428513, 28.0716400, -32.1802979, 25.6516724, -60.8945236, 60.2519302
1: -31.2444191, 25.0189247, -28.6195812, 22.8838177, -54.1282349, 53.6385040
2: -39.8935318, 24.7519379, -36.4810486, 22.6162281, -62.5097580, 61.2329788
3: -42.8789902, 21.0673637, -39.1781273, 19.2350159, -62.1140060, 60.2454834
4: -40.3047600, 28.6806641, -36.9378738, 26.1555099, -66.4602661, 65.6185303
5: -34.9170647, 27.1353035, -31.9317112, 24.8353806, -59.7524452, 59.0670166
6: -32.2504120, 31.7836800, -29.4439697, 29.0753059, -61.3257141, 61.2276382
7: -35.5550537, 33.0544968, -32.4968910, 30.3835773, -65.9386215, 65.5513840
8: -48.5274467, 23.9858456, -44.6601448, 21.6501999, -70.1776428, 68.6459808
9: -31.3520660, 31.5285892, -28.5943222, 28.8028755, -60.1549339, 60.1229057

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0412469, upper bound: 70.0428406
time: 10.67 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0412469, upper bound: 70.0428926
time: 11.68 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -29.3141575, 23.3832932, -25.3644791, 20.3007812, -49.6149368, 48.7477684
1: -26.3353500, 20.8725300, -22.6795464, 18.1351013, -44.4704437, 43.5520706
2: -33.4559212, 20.6033001, -28.8643665, 17.9004536, -51.3563728, 49.4676628
3: -35.8660698, 17.3360291, -30.9543953, 15.1607742, -51.0268402, 48.2904243
4: -34.0581322, 23.8056087, -29.3566551, 20.6620712, -54.7201920, 53.1622620
5: -29.2607250, 22.7262688, -25.2642021, 19.6808891, -48.9416122, 47.9904709
6: -26.8082981, 26.5183296, -23.2334023, 22.9906921, -49.7989883, 49.7517319
7: -29.7435207, 28.2025394, -25.6688766, 24.2817001, -54.0252228, 53.8714142
8: -41.6505508, 18.9820557, -35.8653603, 16.7455425, -58.3960953, 54.8474083
9: -26.0289803, 26.2928314, -22.5525360, 22.7739868, -48.8029671, 48.8453674

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369573, upper bound: 70.0369573
time: 7.73 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369573, upper bound: 70.0369573
time: 12.14 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -29.6767406, 23.6764202, -29.3370781, 23.4208508, -53.0975914, 53.0134964
1: -26.6758461, 21.1390762, -26.1839046, 20.9118004, -47.5876465, 47.3229828
2: -33.8836174, 20.8580284, -33.3415833, 20.6330643, -54.5166817, 54.1996117
3: -36.3304367, 17.5535812, -35.7953339, 17.5011711, -53.8316040, 53.3489151
4: -34.4979515, 24.1027603, -33.8617821, 23.8383675, -58.3363190, 57.9645424
5: -29.6330795, 23.0145607, -29.1856289, 22.7129707, -52.3460503, 52.2001839
6: -27.1448364, 26.8563004, -26.8474541, 26.5487900, -53.6936264, 53.7037544
7: -30.1292419, 28.5670681, -29.6770401, 27.9377556, -58.0669975, 58.2441101
8: -42.1714401, 19.2064991, -41.1499023, 19.4384575, -61.6098862, 60.3563995
9: -26.3622360, 26.6278553, -26.0697250, 26.2850513, -52.6472855, 52.6975784

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369573, upper bound: 70.0369573
time: 9.17 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0369573, upper bound: 70.0369573
time: 7.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.02 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.02
Output dim: 8, lower bound: -70.0412469, upper bound: 70.0428406
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.02
Output dim: 8, lower bound: -70.0412469, upper bound: 70.0428926
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.02
Output dim: 8, lower bound: -70.0412469, upper bound: 70.0428406
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.02
Output dim: 8, lower bound: -70.0412469, upper bound: 70.0428926
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.02
Output dim: 8, lower bound: -70.0369573, upper bound: 70.0369573
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.02
Output dim: 8, lower bound: -70.0369573, upper bound: 70.0369573
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.02
Output dim: 8, lower bound: -70.0369573, upper bound: 70.0369573
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.02
Output dim: 8, lower bound: -70.0369573, upper bound: 70.0369573

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -26.6202583, 21.2837868, -27.7034836, 22.1330051, -48.7532616, 48.9872704
1: -23.7588882, 19.0071583, -24.6833572, 19.7574348, -43.5163193, 43.6905136
2: -30.2525902, 18.7684727, -31.4493027, 19.5230255, -49.7756157, 50.2177734
3: -32.4382744, 15.9198036, -33.7164955, 16.5821533, -49.0204277, 49.6362991
4: -30.7266541, 21.6690903, -31.8983574, 22.5496254, -53.2762794, 53.5674477
5: -26.4748993, 20.6141663, -27.5186977, 21.4222126, -47.8971100, 48.1328659
6: -24.3619461, 24.1155949, -25.3435059, 25.0820732, -49.4440155, 49.4590988
7: -26.9148216, 25.3717041, -27.9882317, 26.3102531, -53.2250748, 53.3599358
8: -37.4352570, 17.6913090, -38.7757416, 18.5338326, -55.9690857, 56.4670486
9: -23.6589355, 23.8746700, -24.6235199, 24.8312340, -48.4901657, 48.4981918

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0454561, upper bound: 70.0463553
time: 9.43 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0454325, upper bound: 70.0463442
time: 9.54 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -31.0836563, 24.7902832, -27.7034836, 22.1330051, -53.2166557, 52.4937668
1: -27.6796188, 22.1215172, -24.6833572, 19.7574348, -47.4370461, 46.8048744
2: -35.2675552, 21.8497868, -31.4493027, 19.5230255, -54.7905807, 53.2990837
3: -37.8789978, 18.5659542, -33.7164955, 16.5821533, -54.4611473, 52.2824478
4: -35.7516823, 25.2560539, -31.8983574, 22.5496254, -58.3013077, 57.1544113
5: -30.8740349, 24.0161915, -27.5186977, 21.4222126, -52.2962341, 51.5348892
6: -28.4426403, 28.1006145, -25.3435059, 25.0820732, -53.5247116, 53.4441223
7: -31.4064445, 29.4421825, -27.9882317, 26.3102531, -57.7166977, 57.4304085
8: -43.3041649, 20.7807980, -38.7757416, 18.5338326, -61.8379898, 59.5565338
9: -27.6168251, 27.8261929, -24.6235199, 24.8312340, -52.4480591, 52.4497147

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 198

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0453307, upper bound: 70.0461054
time: 10.47 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0454325, upper bound: 70.0463442
time: 10.28 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -26.6202583, 21.2837868, -32.1802979, 25.6516724, -52.2719307, 53.4640846
1: -23.7588882, 19.0071583, -28.6195812, 22.8838177, -46.6427078, 47.6267395
2: -30.2525902, 18.7684727, -36.4810486, 22.6162281, -52.8688126, 55.2495193
3: -32.4382744, 15.9198036, -39.1781273, 19.2350159, -51.6732864, 55.0979309
4: -30.7266541, 21.6690903, -36.9378738, 26.1555099, -56.8821640, 58.6069603
5: -26.4748993, 20.6141663, -31.9317112, 24.8353806, -51.3102798, 52.5458755
6: -24.3619461, 24.1155949, -29.4439697, 29.0753059, -53.4372520, 53.5595627
7: -26.9148216, 25.3717041, -32.4968910, 30.3835773, -57.2983932, 57.8685913
8: -37.4352570, 17.6913090, -44.6601448, 21.6501999, -59.0854568, 62.3514557
9: -23.6589355, 23.8746700, -28.5943222, 28.8028755, -52.4618034, 52.4689903

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 80

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0412469, upper bound: 70.0428406
time: 10.51 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0412407, upper bound: 70.0428352
time: 8.60 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -31.0836563, 24.7902832, -32.1802979, 25.6516724, -56.7353287, 56.9705734
1: -27.6796188, 22.1215172, -28.6195812, 22.8838177, -50.5634384, 50.7410965
2: -35.2675552, 21.8497868, -36.4810486, 22.6162281, -57.8837814, 58.3308334
3: -37.8789978, 18.5659542, -39.1781273, 19.2350159, -57.1139984, 57.7440758
4: -35.7516823, 25.2560539, -36.9378738, 26.1555099, -61.9071922, 62.1939278
5: -30.8740349, 24.0161915, -31.9317112, 24.8353806, -55.7094154, 55.9479027
6: -28.4426403, 28.1006145, -29.4439697, 29.0753059, -57.5179443, 57.5445862
7: -31.4064445, 29.4421825, -32.4968910, 30.3835773, -61.7900238, 61.9390717
8: -43.3041649, 20.7807980, -44.6601448, 21.6501999, -64.9543610, 65.4409409
9: -27.6168251, 27.8261929, -28.5943222, 28.8028755, -56.4196968, 56.4205132

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0387857, upper bound: 70.0400171
time: 10.27 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0371273, upper bound: 70.0387129
time: 9.37 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -23.8933449, 19.1247635, -25.3644791, 20.3007812, -44.1941261, 44.4892387
1: -21.5397129, 17.0840530, -22.6795464, 18.1351013, -39.6748085, 39.7635956
2: -27.3195496, 16.8319569, -28.8643665, 17.9004536, -45.2200012, 45.6963234
3: -29.2478065, 14.1104231, -30.9543953, 15.1607742, -44.4085808, 45.0648193
4: -27.8731976, 19.4391842, -29.3566551, 20.6620712, -48.5352631, 48.7958374
5: -23.9226494, 18.6305428, -25.2642021, 19.6808891, -43.6035309, 43.8947334
6: -21.8609161, 21.6355057, -23.2334023, 22.9906921, -44.8516083, 44.8689079
7: -24.2307262, 23.2052383, -25.6688766, 24.2817001, -48.5124207, 48.8741150
8: -34.4005241, 15.2866468, -35.8653603, 16.7455425, -51.1460533, 51.1520081
9: -21.2250462, 21.4622612, -22.5525360, 22.7739868, -43.9990311, 44.0147972

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0383954, upper bound: 70.0379320
time: 18.65 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0387142, upper bound: 70.0381409
time: 16.05 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -26.5051231, 21.2003975, -25.3644791, 20.3007812, -46.8059044, 46.5648766
1: -23.9098358, 18.9253578, -22.6795464, 18.1351013, -42.0449295, 41.6049042
2: -30.3303452, 18.6445999, -28.8643665, 17.9004536, -48.2307968, 47.5089645
3: -32.4786453, 15.6348019, -30.9543953, 15.1607742, -47.6394196, 46.5891953
4: -30.9363499, 21.5362873, -29.3566551, 20.6620712, -51.5984192, 50.8929443
5: -26.5438118, 20.6512108, -25.2642021, 19.6808891, -46.2247009, 45.9154091
6: -24.2591343, 24.0123444, -23.2334023, 22.9906921, -47.2498245, 47.2457466
7: -26.9261360, 25.7189884, -25.6688766, 24.2817001, -51.2078362, 51.3878632
8: -38.0609360, 16.9489403, -35.8653603, 16.7455425, -54.8064766, 52.8143005
9: -23.5536003, 23.8045216, -22.5525360, 22.7739868, -46.3275871, 46.3570557

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0383954, upper bound: 70.0379320
time: 7.71 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0387142, upper bound: 70.0381409
time: 9.78 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -23.8933449, 19.1247635, -29.3370781, 23.4208508, -47.3141937, 48.4618340
1: -21.5397129, 17.0840530, -26.1839046, 20.9118004, -42.4515076, 43.2679558
2: -27.3195496, 16.8319569, -33.3415833, 20.6330643, -47.9526138, 50.1735382
3: -29.2478065, 14.1104231, -35.7953339, 17.5011711, -46.7489777, 49.9057579
4: -27.8731976, 19.4391842, -33.8617821, 23.8383675, -51.7115631, 53.3009644
5: -23.9226494, 18.6305428, -29.1856289, 22.7129707, -46.6356087, 47.8161697
6: -21.8609161, 21.6355057, -26.8474541, 26.5487900, -48.4097061, 48.4829597
7: -24.2307262, 23.2052383, -29.6770401, 27.9377556, -52.1684723, 52.8822784
8: -34.4005241, 15.2866468, -41.1499023, 19.4384575, -53.8389740, 56.4365501
9: -21.2250462, 21.4622612, -26.0697250, 26.2850513, -47.5100937, 47.5319824

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0368028, upper bound: 70.0367874
time: 9.50 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0368028, upper bound: 70.0369573
time: 7.21 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -26.5051231, 21.2003975, -29.3370781, 23.4208508, -49.9259720, 50.5374756
1: -23.9098358, 18.9253578, -26.1839046, 20.9118004, -44.8216324, 45.1092606
2: -30.3303452, 18.6445999, -33.3415833, 20.6330643, -50.9634094, 51.9861832
3: -32.4786453, 15.6348019, -35.7953339, 17.5011711, -49.9798164, 51.4301376
4: -30.9363499, 21.5362873, -33.8617821, 23.8383675, -54.7747192, 55.3980713
5: -26.5438118, 20.6512108, -29.1856289, 22.7129707, -49.2567787, 49.8368378
6: -24.2591343, 24.0123444, -26.8474541, 26.5487900, -50.8079224, 50.8597984
7: -26.9261360, 25.7189884, -29.6770401, 27.9377556, -54.8638840, 55.3960266
8: -38.0609360, 16.9489403, -41.1499023, 19.4384575, -57.4993935, 58.0988426
9: -23.5536003, 23.8045216, -26.0697250, 26.2850513, -49.8386536, 49.8742409

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0277471, upper bound: 70.0273403
time: 9.04 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0265943, upper bound: 70.0265943
time: 6.38 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 16.77 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 8, lower bound: -70.0454561, upper bound: 70.0463553
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 8, lower bound: -70.0454325, upper bound: 70.0463442
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 8, lower bound: -70.0453307, upper bound: 70.0461054
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 8, lower bound: -70.0454325, upper bound: 70.0463442
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 8, lower bound: -70.0412469, upper bound: 70.0428406
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 8, lower bound: -70.0412407, upper bound: 70.0428352
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 8, lower bound: -70.0387857, upper bound: 70.0400171
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 8, lower bound: -70.0371273, upper bound: 70.0387129
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 8, lower bound: -70.0383954, upper bound: 70.0379320
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 8, lower bound: -70.0387142, upper bound: 70.0381409
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 8, lower bound: -70.0383954, upper bound: 70.0379320
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 8, lower bound: -70.0387142, upper bound: 70.0381409
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 8, lower bound: -70.0368028, upper bound: 70.0367874
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 8, lower bound: -70.0368028, upper bound: 70.0369573
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 8, lower bound: -70.0277471, upper bound: 70.0273403
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 8, lower bound: -70.0265943, upper bound: 70.0265943

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -22.9036407, 18.3686848, -27.0360813, 21.6090126, -44.5126419, 45.4047661
1: -20.5226936, 16.4367752, -24.0971260, 19.2958965, -39.8185883, 40.5338974
2: -26.1114540, 16.2047539, -30.6991463, 19.0593262, -45.1707764, 46.9038925
3: -28.0204697, 13.6678467, -32.9125023, 16.1852646, -44.2057304, 46.5803490
4: -26.6302509, 18.7065811, -31.1525879, 22.0123558, -48.6426048, 49.8591690
5: -22.8820438, 17.8519630, -26.8655567, 20.9213734, -43.8034172, 44.7175217
6: -21.0279617, 20.7833920, -24.7372360, 24.4853287, -45.5132904, 45.5206223
7: -23.2144928, 22.0983067, -27.3176651, 25.7055702, -48.9200630, 49.4159698
8: -32.6424675, 14.9692726, -37.8957100, 18.0601654, -50.7026291, 52.8649826
9: -20.4006653, 20.5874977, -24.0335007, 24.2401466, -44.6408119, 44.6209984

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 80

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0461726, upper bound: 70.0470609
time: 10.03 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0455256, upper bound: 70.0466600
time: 10.79 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -25.6333103, 20.5103760, -27.3075027, 21.8223286, -47.4556389, 47.8178787
1: -22.8887177, 18.3244476, -24.3354816, 19.4839458, -42.3726654, 42.6599274
2: -29.1415825, 18.0875664, -31.0038757, 19.2485809, -48.3901558, 49.0914345
3: -31.2511673, 15.3407211, -33.2390099, 16.3481903, -47.5993576, 48.5797234
4: -29.6184387, 20.8772240, -31.4552193, 22.2304955, -51.8489304, 52.3324394
5: -25.5085869, 19.8737640, -27.1307373, 21.1249409, -46.6335297, 47.0044937
6: -23.4689255, 23.2323189, -24.9836311, 24.7282429, -48.1971664, 48.2159500
7: -25.9228897, 24.4752922, -27.5902729, 25.9514656, -51.8743553, 52.0655670
8: -36.1287994, 17.0060139, -38.2542419, 18.2542820, -54.3830757, 55.2602539
9: -22.7900410, 23.0039749, -24.2736855, 24.4808617, -47.2708969, 47.2776604

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 80

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0460566, upper bound: 70.0468669
time: 9.97 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0453009, upper bound: 70.0463398
time: 11.96 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -30.4191589, 24.2686234, -23.8659058, 19.1252174, -49.5443764, 48.1345291
1: -27.0951958, 21.6624050, -21.3460999, 17.1056118, -44.2008057, 43.0085030
2: -34.5225182, 21.3873634, -27.1767673, 16.8748322, -51.3973389, 48.5641251
3: -37.0787277, 18.1721649, -29.1628494, 14.2504892, -51.3292160, 47.3350143
4: -35.0103035, 24.7207317, -27.6796398, 19.4762459, -54.4865494, 52.4003716
5: -30.2247810, 23.5178413, -23.8079491, 18.5714531, -48.7962341, 47.3257904
6: -27.8377419, 27.5071030, -21.9003181, 21.6456375, -49.4833794, 49.4074211
7: -30.7389297, 28.8432751, -24.1665955, 22.9352989, -53.6742287, 53.0098724
8: -42.4308777, 20.3069458, -33.8504601, 15.6983175, -58.1291962, 54.1574020
9: -27.0298386, 27.2374687, -21.2537689, 21.4351387, -48.4649734, 48.4912376

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0426196, upper bound: 70.0434046
time: 11.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0419753, upper bound: 70.0426315
time: 14.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -30.6730881, 24.4683666, -26.6851883, 21.3340092, -52.0070953, 51.1535416
1: -27.3186703, 21.8379803, -23.7879105, 19.0531521, -46.3718224, 45.6258850
2: -34.8067627, 21.5651932, -30.3033180, 18.8177872, -53.6245461, 51.8685112
3: -37.3845406, 18.3238029, -32.4894753, 15.9799585, -53.3644943, 50.8132782
4: -35.2930260, 24.9251766, -30.7579536, 21.7290630, -57.0220871, 55.6831284
5: -30.4725380, 23.7082024, -26.5210438, 20.6575165, -51.1300545, 50.2292442
6: -28.0691433, 27.7341366, -24.4191017, 24.1712837, -52.2404251, 52.1532364
7: -30.9941692, 29.0718365, -26.9650383, 25.3860855, -56.3802567, 56.0368729
8: -42.7648888, 20.4899101, -37.4332047, 17.8166790, -60.5815659, 57.9231110
9: -27.2543755, 27.4628429, -23.7237167, 23.9298115, -51.1841850, 51.1865616

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0238373, upper bound: 70.0252388
time: 10.91 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0235612, upper bound: 70.0250396
time: 12.26 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -22.9036407, 18.3686848, -31.4924583, 25.1113701, -48.0150070, 49.8611450
1: -20.5226936, 16.4367752, -28.0172005, 22.4073219, -42.9300156, 44.4539680
2: -26.1114540, 16.2047539, -35.7101555, 22.1380978, -48.2495499, 51.9149017
3: -28.0204697, 13.6678467, -38.3507843, 18.8270302, -46.8474960, 52.0186310
4: -26.6302509, 18.7065811, -36.1711464, 25.5997467, -52.2299919, 54.8777237
5: -22.8820438, 17.8519630, -31.2603016, 24.3188934, -47.2009354, 49.1122589
6: -21.0279617, 20.7833920, -28.8179760, 28.4622002, -49.4901619, 49.6013680
7: -23.2144928, 22.0983067, -31.8072605, 29.7662926, -52.9807777, 53.9055634
8: -32.6424675, 14.9692726, -43.7575760, 21.1535244, -53.7959900, 58.7268486
9: -20.4006653, 20.5874977, -27.9862480, 28.1916637, -48.5923309, 48.5737457

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 80

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0390737, upper bound: 70.0402512
time: 9.12 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0373832, upper bound: 70.0389428
time: 10.83 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -25.6333103, 20.5103760, -31.7536354, 25.3171062, -50.9504166, 52.2640114
1: -22.8887177, 18.3244476, -28.2465458, 22.5884209, -45.4771347, 46.5709915
2: -29.1415825, 18.0875664, -36.0026817, 22.3209229, -51.4625053, 54.0902405
3: -31.2511673, 15.3407211, -38.6650276, 18.9835262, -50.2346916, 54.0057411
4: -29.6184387, 20.8772240, -36.4617462, 25.8105488, -55.4289856, 57.3389664
5: -25.5085869, 19.8737640, -31.5152206, 24.5149574, -50.0235443, 51.3889847
6: -23.4689255, 23.2323189, -29.0558891, 28.6955090, -52.1644287, 52.2882080
7: -25.9228897, 24.4752922, -32.0696182, 30.0007973, -55.9236870, 56.5449066
8: -36.1287994, 17.0060139, -44.1009750, 21.3436756, -57.4724731, 61.1069870
9: -22.7900410, 23.0039749, -28.2171535, 28.4243317, -51.2143707, 51.2211304

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 80

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0390718, upper bound: 70.0402491
time: 9.66 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0373797, upper bound: 70.0389418
time: 10.42 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -30.5637894, 24.3848629, -30.6713352, 24.4740524, -55.0378342, 55.0561981
1: -27.2258148, 21.7606564, -27.3061256, 21.8331375, -49.0589523, 49.0667763
2: -34.6911201, 21.4843006, -34.8056984, 21.5560188, -56.2471390, 56.2900009
3: -37.2590981, 18.2535915, -37.3802109, 18.3251591, -55.5842590, 55.6338043
4: -35.1803246, 24.8341980, -35.2799683, 24.9274406, -60.1077652, 60.1141663
5: -30.3712349, 23.6318092, -30.4711876, 23.7177200, -54.0889549, 54.1029930
6: -27.9716415, 27.6372910, -28.0755959, 27.7323093, -55.7039413, 55.7128868
7: -30.8862534, 28.9882717, -30.9872379, 29.0692291, -59.9554672, 59.9755096
8: -42.6420555, 20.3914013, -42.7393875, 20.5060844, -63.1481323, 63.1307907
9: -27.1595173, 27.3660641, -27.2662354, 27.4623585, -54.6218719, 54.6323013

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0371664, upper bound: 70.0387129
time: 9.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0371664, upper bound: 70.0387129
time: 9.46 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -30.3196316, 24.1943569, -33.8331490, 26.9714317, -57.2910614, 58.0275040
1: -27.0151863, 21.5894585, -30.1337051, 24.0348454, -51.0500336, 51.7231522
2: -34.4252396, 21.3110104, -38.4438782, 23.7475338, -58.1727676, 59.7548866
3: -36.9694443, 18.0998898, -41.2264977, 20.1497726, -57.1192169, 59.3263817
4: -34.9207001, 24.6322765, -38.9315491, 27.4957790, -62.4164772, 63.5638275
5: -30.1392021, 23.4535580, -33.6039658, 26.1256065, -56.2648087, 57.0575180
6: -27.7491550, 27.4192696, -30.9441319, 30.5789375, -58.3280869, 58.3634033
7: -30.6433563, 28.7855186, -34.2123604, 32.0176010, -62.6609421, 62.9978790
8: -42.3449783, 20.1936398, -47.0670624, 22.6634483, -65.0084152, 67.2607040
9: -26.9447384, 27.1512318, -30.0943279, 30.3326874, -57.2774124, 57.2455597

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 183

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0371664, upper bound: 70.0387129
time: 8.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0371664, upper bound: 70.0387129
time: 10.89 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -21.4321442, 17.1874352, -24.7397499, 19.8089600, -41.2411041, 41.9271812
1: -19.3396912, 15.3591290, -22.1282806, 17.7020073, -37.0416985, 37.4874115
2: -24.5420513, 15.1124277, -28.1589394, 17.4669571, -42.0090027, 43.2713623
3: -26.2638836, 12.6253223, -30.2008362, 14.7909727, -41.0548553, 42.8261566
4: -25.0854702, 17.4663696, -28.6542168, 20.1637917, -45.2492599, 46.1205826
5: -21.5097332, 16.7671204, -24.6534481, 19.2103653, -40.7201004, 41.4205704
6: -19.6275539, 19.4216881, -22.6658859, 22.4301262, -42.0576782, 42.0875702
7: -21.7308350, 20.9347534, -25.0409698, 23.7127190, -45.4435539, 45.9757195
8: -31.0741196, 13.5872135, -35.0328445, 16.3095322, -47.3836479, 48.6200562
9: -19.0463829, 19.2807865, -22.0016022, 22.2202168, -41.2666016, 41.2823868

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 80

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0238925, upper bound: 70.0232013
time: 11.56 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0216309, upper bound: 70.0216185
time: 7.64 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -22.9441776, 18.3775368, -24.9941883, 20.0093307, -42.9535065, 43.3717270
1: -20.6926727, 16.4218407, -22.3527641, 17.8783913, -38.5710640, 38.7746048
2: -26.2371178, 16.1739235, -28.4457569, 17.6442795, -43.8813934, 44.6196823
3: -28.0900593, 13.5546455, -30.5074425, 14.9426746, -43.0327339, 44.0620880
4: -26.7850952, 18.6769085, -28.9393539, 20.3664589, -47.1515541, 47.6162643
5: -22.9849472, 17.9106979, -24.9017239, 19.4018250, -42.3867722, 42.8124237
6: -20.9965038, 20.7820950, -22.8972149, 22.6583443, -43.6548462, 43.6793022
7: -23.2651863, 22.3155155, -25.2965508, 23.9442387, -47.2094231, 47.6120682
8: -33.1002884, 14.6577711, -35.3722305, 16.4889107, -49.5891876, 50.0299988
9: -20.3836498, 20.6226482, -22.2260895, 22.4458580, -42.8295021, 42.8487320

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 80

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0245484, upper bound: 70.0238050
time: 9.71 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -70.0221347, upper bound: 70.0221346
time: 11.91 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -23.9489746, 19.1936646, -24.7397499, 19.8089600, -43.7579269, 43.9334145
1: -21.6315460, 17.1339607, -22.1282806, 17.7020073, -39.3335495, 39.2622414
2: -27.4474907, 16.8601990, -28.1589394, 17.4669571, -44.9144440, 45.0191269
3: -29.3747540, 14.0999355, -30.2008362, 14.7909727, -44.1657257, 44.3007736
4: -28.0342941, 19.4970894, -28.6542168, 20.1637917, -48.1980858, 48.1513062
5: -24.0379372, 18.7217216, -24.6534481, 19.2103653, -43.2483025, 43.3751640
6: -21.9494476, 21.7117195, -22.6658859, 22.4301262, -44.3795738, 44.3776054
7: -24.3309727, 23.3641510, -25.0409698, 23.7127190, -48.0436935, 48.4051094
8: -34.6134300, 15.2006092, -35.0328445, 16.3095322, -50.9229507, 50.2334518
9: -21.2993031, 21.5366611, -22.0016022, 22.2202168, -43.5195198, 43.5382538

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0277207, upper bound: 70.0269413
time: 9.77 seconds

## Relational analysis of IS_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0265907, upper bound: 70.0260901
time: 8.94 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -25.5266571, 20.4296455, -24.9941883, 20.0093307, -45.5359802, 45.4238281
1: -23.0375404, 18.2442284, -22.3527641, 17.8783913, -40.9159317, 40.5969925
2: -29.2174377, 17.9645653, -28.4457569, 17.6442795, -46.8617096, 46.4103203
3: -31.2880478, 15.0604839, -30.5074425, 14.9426746, -46.2307205, 45.5679207
4: -29.8162975, 20.7529068, -28.9393539, 20.3664589, -50.1827545, 49.6922607
5: -25.5791817, 19.9101524, -24.9017239, 19.4018250, -44.9810028, 44.8118744
6: -23.3693295, 23.1315994, -22.8972149, 22.6583443, -46.0276718, 46.0288162
7: -25.9303513, 24.8067398, -25.2965508, 23.9442387, -49.8745880, 50.1032906
8: -36.7261963, 16.2970409, -35.3722305, 16.4889107, -53.2151070, 51.6692734
9: -22.6894970, 22.9333458, -22.2260895, 22.4458580, -45.1353493, 45.1594238

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0293846, upper bound: 70.0284818
time: 8.92 seconds

## Relational analysis of IS_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0280461, upper bound: 70.0275466
time: 7.95 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.21 seconds
IS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0461726, upper bound: 70.0470609
IS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0455256, upper bound: 70.0466600
IS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0460566, upper bound: 70.0468669
IS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0453009, upper bound: 70.0463398
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0426196, upper bound: 70.0434046
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0419753, upper bound: 70.0426315
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0238373, upper bound: 70.0252388
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0235612, upper bound: 70.0250396
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0390737, upper bound: 70.0402512
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0373832, upper bound: 70.0389428
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0390718, upper bound: 70.0402491
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0373797, upper bound: 70.0389418
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0371664, upper bound: 70.0387129
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0371664, upper bound: 70.0387129
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0371664, upper bound: 70.0387129
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0371664, upper bound: 70.0387129
IS_A2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0238925, upper bound: 70.0232013
IS_A2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0216309, upper bound: 70.0216185
IS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0245484, upper bound: 70.0238050
IS_A2_B1_A1_A2_B2, status: Status.VERIFIED, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0221347, upper bound: 70.0221346
IS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0277207, upper bound: 70.0269413
IS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0265907, upper bound: 70.0260901
IS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0293846, upper bound: 70.0284818
IS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.21
Output dim: 8, lower bound: -70.0280461, upper bound: 70.0275466
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 18.21
Output dim: 8, lower bound: -70.0368028, upper bound: 70.0367874
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 18.21
Output dim: 8, lower bound: -70.0368028, upper bound: 70.0369573
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.21
Output dim: 8, lower bound: -70.0277471, upper bound: 70.0273403
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.21
Output dim: 8, lower bound: -70.0265943, upper bound: 70.0265943
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=82.72178649902344
rel_dist={8: [-70.08157121987698, 70.08157121987699]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0730386, upper bound: 70.0730084
time: 9.91 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0735878, upper bound: 70.0735878
time: 9.04 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.10 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.10
Output dim: 8, lower bound: -70.0730386, upper bound: 70.0730084
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.10
Output dim: 8, lower bound: -70.0735878, upper bound: 70.0735878

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -27.7034836, 22.1330051, -32.2284393, 25.6844826, -53.3879662, 54.3614349
1: -24.6833572, 19.7574348, -28.6379700, 22.8967686, -47.5801239, 48.3954010
2: -31.4493027, 19.5230255, -36.5253944, 22.6556149, -54.1049042, 56.0484200
3: -33.7164955, 16.5821533, -39.1806412, 19.2624130, -52.9789085, 55.7627869
4: -31.8983574, 22.5496254, -36.9531784, 26.2139816, -58.1123390, 59.5028038
5: -27.5186977, 21.4222126, -31.9482155, 24.8399353, -52.3586235, 53.3704224
6: -25.3435059, 25.0820732, -29.4757481, 29.1102676, -54.4537735, 54.5578232
7: -27.9882317, 26.3102531, -32.5274353, 30.3702316, -58.3584633, 58.8376884
8: -38.7757416, 18.5338326, -44.6720009, 21.7884808, -60.5642242, 63.2058334
9: -24.6235199, 24.8312340, -28.6361542, 28.8533592, -53.4768791, 53.4673843

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 80

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0716204, upper bound: 70.0716461
time: 8.90 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0715368, upper bound: 70.0715194
time: 11.67 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -32.1802979, 25.6516724, -33.8552818, 26.9734821, -59.1537781, 59.5069542
1: -28.6195812, 22.8838177, -30.0531197, 24.0491009, -52.6686821, 52.9369354
2: -36.4810486, 22.6162281, -38.3427620, 23.7854748, -60.2665253, 60.9589844
3: -39.1781273, 19.2350159, -41.1858978, 20.2414703, -59.4195900, 60.4209061
4: -36.9378738, 26.1555099, -38.7724915, 27.5361214, -64.4739838, 64.9280014
5: -31.9317112, 24.8353806, -33.5569725, 26.0933819, -58.0250931, 58.3923531
6: -29.4439697, 29.0753059, -30.9779434, 30.5584316, -60.0024033, 60.0532379
7: -32.4968910, 30.3835773, -34.1661568, 31.8342838, -64.3311691, 64.5497360
8: -44.6601448, 21.6501999, -46.7624969, 22.9480820, -67.6082306, 68.4126968
9: -28.5943222, 28.8028755, -30.1012688, 30.2933197, -58.8876381, 58.9041443

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0596694, upper bound: 70.0595221
time: 10.39 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0591185, upper bound: 70.0591184
time: 8.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 20.31 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 20.31
Output dim: 8, lower bound: -70.0716204, upper bound: 70.0716461
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 20.31
Output dim: 8, lower bound: -70.0715368, upper bound: 70.0715194
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.31
Output dim: 8, lower bound: -70.0596694, upper bound: 70.0595221
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.31
Output dim: 8, lower bound: -70.0591185, upper bound: 70.0591184

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -23.8659058, 19.1252174, -30.9900875, 24.7124004, -48.5783043, 50.1153030
1: -21.3460999, 17.1056118, -27.5555534, 22.0392609, -43.3853531, 44.6611595
2: -27.1767673, 16.8748322, -35.1387901, 21.7960968, -48.9728622, 52.0136070
3: -29.1628494, 14.2504892, -37.6919174, 18.5300198, -47.6928711, 51.9424057
4: -27.6796398, 19.4762459, -35.5698853, 25.2129116, -52.8925514, 55.0461273
5: -23.8079491, 18.5714531, -30.7402802, 23.9096565, -47.7176056, 49.3117332
6: -21.9003181, 21.6456375, -28.3482265, 28.0074692, -49.9077835, 49.9938660
7: -24.1665955, 22.9352989, -31.2852039, 29.2616405, -53.4282379, 54.2205048
8: -33.8504601, 15.6983175, -43.0483093, 20.8935051, -54.7439651, 58.7466278
9: -21.2537689, 21.4351387, -27.5392437, 27.7555237, -49.0092926, 48.9743767

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0665098, upper bound: 70.0664746
time: 9.35 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0668172, upper bound: 70.0668161
time: 9.27 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -26.6851883, 21.3340092, -31.3697300, 25.0106621, -51.6958466, 52.7037392
1: -23.7879105, 19.0531521, -27.8877373, 22.3018913, -46.0897980, 46.9408875
2: -30.3033180, 18.8177872, -35.5626373, 22.0613098, -52.3646278, 54.3804207
3: -32.4894753, 15.9799585, -38.1476288, 18.7568321, -51.2463074, 54.1275826
4: -30.7579536, 21.7290630, -35.9927597, 25.5188408, -56.2767944, 57.7218208
5: -26.5210438, 20.6575165, -31.1101265, 24.1939068, -50.7149506, 51.7676430
6: -24.4191017, 24.1712837, -28.6938095, 28.3457832, -52.7648849, 52.8650932
7: -26.9650383, 25.3860855, -31.6662636, 29.6008701, -56.5659065, 57.0523491
8: -37.4332047, 17.8166790, -43.5463600, 21.1706734, -58.6038742, 61.3630371
9: -23.7237167, 23.9298115, -27.8751450, 28.0922756, -51.8159943, 51.8049545

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0664344, upper bound: 70.0663625
time: 11.14 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0667579, upper bound: 70.0666942
time: 10.62 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -29.4961014, 23.5332146, -29.7885761, 23.7594604, -53.2555542, 53.3217926
1: -26.2845688, 21.0095253, -26.5289268, 21.2074242, -47.4919930, 47.5384521
2: -33.4704742, 20.7421970, -33.7874908, 20.9480019, -54.4184723, 54.5296860
3: -35.9440956, 17.6087265, -36.2926445, 17.7869339, -53.7310219, 53.9013710
4: -33.9791107, 23.9567451, -34.2953224, 24.1950531, -58.1741638, 58.2520638
5: -29.3161812, 22.8317242, -29.5978241, 23.0543289, -52.3705063, 52.4295464
6: -26.9831200, 26.6626301, -27.2473946, 26.9104309, -53.8935509, 53.9100189
7: -29.7970448, 28.0163536, -30.0801201, 28.2673130, -58.0643501, 58.0964661
8: -41.2644272, 19.5985756, -41.6273003, 19.8141670, -61.0785904, 61.2258759
9: -26.1727562, 26.4087105, -26.4222641, 26.6648331, -52.8375893, 52.8309669

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0590707, upper bound: 70.0589619
time: 10.30 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0590693, upper bound: 70.0589436
time: 8.19 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -25.9979362, 20.7880821, -28.0690708, 22.4085655, -48.4064980, 48.8571434
1: -23.2424011, 18.5715332, -25.0641880, 20.0062065, -43.2486038, 43.6357193
2: -29.5505695, 18.3308125, -31.8613453, 19.7897987, -49.3403702, 50.1921539
3: -31.7257881, 15.5051479, -34.2132835, 16.7482986, -48.4740868, 49.7184296
4: -30.1197891, 21.1448517, -32.4301376, 22.8136864, -52.9334755, 53.5749893
5: -25.9172840, 20.2224731, -27.9210663, 21.7916603, -47.7089462, 48.1435394
6: -23.8203735, 23.5237236, -25.6997757, 25.3775139, -49.1978874, 49.2234955
7: -26.2911301, 24.9372482, -28.3495026, 26.8041153, -53.0952454, 53.2867508
8: -36.8111649, 17.0365162, -39.5785789, 18.5172806, -55.3284454, 56.6150970
9: -23.0679970, 23.3357697, -24.8849182, 25.1832886, -48.2512856, 48.2206841

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 198

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0586387, upper bound: 70.0586584
time: 9.90 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0586096, upper bound: 70.0586096
time: 9.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.02 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 21.02
Output dim: 8, lower bound: -70.0665098, upper bound: 70.0664746
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 21.02
Output dim: 8, lower bound: -70.0668172, upper bound: 70.0668161
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 21.02
Output dim: 8, lower bound: -70.0664344, upper bound: 70.0663625
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 21.02
Output dim: 8, lower bound: -70.0667579, upper bound: 70.0666942
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.02
Output dim: 8, lower bound: -70.0590707, upper bound: 70.0589619
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.02
Output dim: 8, lower bound: -70.0590693, upper bound: 70.0589436
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.02
Output dim: 8, lower bound: -70.0586387, upper bound: 70.0586584
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.02
Output dim: 8, lower bound: -70.0586096, upper bound: 70.0586096

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -20.0978546, 16.1550922, -23.7860298, 19.0614815, -39.1593323, 39.9411163
1: -18.0237656, 14.4653683, -21.2471161, 17.0329437, -35.0567093, 35.7124863
2: -22.9180717, 14.2654438, -27.0371437, 16.8287582, -39.7468300, 41.3025894
3: -24.5594978, 11.9915123, -28.9983120, 14.2264585, -38.7859573, 40.9898186
4: -23.4319038, 16.4421082, -27.5282021, 19.3901138, -42.8220177, 43.9703102
5: -20.0819626, 15.7305164, -23.6663189, 18.4943218, -38.5762787, 39.3968353
6: -18.4792347, 18.2703400, -21.8257141, 21.5943832, -40.0736160, 40.0960541
7: -20.3795605, 19.4878998, -24.0757370, 22.7882576, -43.1678123, 43.5636368
8: -28.8560791, 13.0729942, -33.6704407, 15.7370234, -44.5930939, 46.7434349
9: -17.9101067, 18.0828190, -21.1482296, 21.3649902, -39.2750893, 39.2310410

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 242

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0564503, upper bound: 70.0563660
time: 9.80 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0566293, upper bound: 70.0566093
time: 10.36 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -21.1469307, 16.9858475, -26.4843750, 21.1704159, -42.3173447, 43.4702225
1: -18.9684696, 15.2047558, -23.6522369, 18.9091454, -37.8776169, 38.8569870
2: -24.1206589, 14.9910011, -30.1026859, 18.6764851, -42.7971420, 45.0936813
3: -25.8709183, 12.6175365, -32.2768135, 15.8018341, -41.6727524, 44.8943481
4: -24.6425819, 17.2920876, -30.5957623, 21.5527687, -46.1953430, 47.8878479
5: -21.1339779, 16.5299358, -26.3260727, 20.5247822, -41.6587601, 42.8560104
6: -19.4404202, 19.2184715, -24.2633438, 24.0113029, -43.4517212, 43.4818077
7: -21.4499531, 20.4738617, -26.8031673, 25.2688446, -46.7187958, 47.2770309
8: -30.2896309, 13.7721539, -37.2857590, 17.5565643, -47.8461952, 51.0579147
9: -18.8457584, 19.0212269, -23.5348625, 23.7486572, -42.5944138, 42.5560837

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 112

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0571666, upper bound: 70.0571855
time: 8.86 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0571743, upper bound: 70.0572364
time: 10.50 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -22.4974270, 18.0423260, -24.1420803, 19.3417721, -41.8391991, 42.1844063
1: -20.1235409, 16.1348476, -21.5624313, 17.2797184, -37.4032516, 37.6972733
2: -25.5967102, 15.9356174, -27.4374046, 17.0770874, -42.6737862, 43.3730125
3: -27.4429703, 13.4681787, -29.4273758, 14.4386406, -41.8816109, 42.8955460
4: -26.0849571, 18.3594303, -27.9266052, 19.6741543, -45.7591095, 46.2860336
5: -22.4027023, 17.5135689, -24.0141525, 18.7618313, -41.1645317, 41.5277138
6: -20.6455269, 20.4351597, -22.1494675, 21.9135704, -42.5590973, 42.5846214
7: -22.7844257, 21.6168060, -24.4331627, 23.1124783, -45.8969040, 46.0499611
8: -31.9868183, 14.8293800, -34.1460495, 15.9884357, -47.9752502, 48.9754257
9: -20.0091019, 20.2222691, -21.4622135, 21.6802807, -41.6893692, 41.6844826

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0563693, upper bound: 70.0562448
time: 9.84 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0565469, upper bound: 70.0564702
time: 9.30 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -23.5186462, 18.8474770, -26.8329506, 21.4441452, -44.9627838, 45.6804276
1: -21.0333176, 16.8509750, -23.9611912, 19.1502056, -40.1835213, 40.8121643
2: -26.7617149, 16.6371918, -30.4935665, 18.9205112, -45.6822128, 47.1307602
3: -28.6987591, 14.0728970, -32.6953735, 16.0085850, -44.7073402, 46.7682648
4: -27.2472534, 19.1822815, -30.9847488, 21.8326187, -49.0798645, 50.1670303
5: -23.4163628, 18.2845001, -26.6661797, 20.7856121, -44.2019653, 44.9506798
6: -21.5732918, 21.3526840, -24.5800400, 24.3229370, -45.8962288, 45.9327240
7: -23.8186302, 22.5588112, -27.1543770, 25.5841293, -49.4027596, 49.7131882
8: -33.3524132, 15.5306282, -37.7471161, 17.8060799, -51.1584930, 53.2777443
9: -20.9208088, 21.1293392, -23.8423214, 24.0576324, -44.9784279, 44.9716568

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0570755, upper bound: 70.0570206
time: 10.92 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0570871, upper bound: 70.0570761
time: 9.15 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -25.7436314, 20.6033363, -28.6499290, 22.8676109, -48.6112442, 49.2532654
1: -23.0249939, 18.4225502, -25.5281048, 20.4209824, -43.4459763, 43.9506531
2: -29.3063831, 18.1585655, -32.5100021, 20.1567459, -49.4631271, 50.6685677
3: -31.4686069, 15.3370457, -34.9207764, 17.1123581, -48.5809631, 50.2578125
4: -29.8639050, 20.9619961, -33.0246658, 23.2803841, -53.1442871, 53.9866638
5: -25.6958618, 20.0475121, -28.4869995, 22.2004948, -47.8963547, 48.5345078
6: -23.6194687, 23.3168564, -26.2134514, 25.8943157, -49.5137863, 49.5303040
7: -26.0651894, 24.7342949, -28.9375610, 27.2389526, -53.3041420, 53.6718407
8: -36.4639664, 16.8551445, -40.1282654, 19.0068340, -55.4707985, 56.9834099
9: -22.8981152, 23.1020851, -25.4192257, 25.6583424, -48.5564575, 48.5213051

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 83

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0483392, upper bound: 70.0481194
time: 13.03 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0482089, upper bound: 70.0480216
time: 10.03 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -28.4590187, 22.7217712, -28.9969921, 23.1404343, -51.5994530, 51.7187653
1: -25.3720856, 20.2932739, -25.8332024, 20.6609955, -46.0330811, 46.1264725
2: -32.3050270, 20.0250397, -32.8984985, 20.3999348, -52.7049599, 52.9235382
3: -34.6937599, 16.9972572, -35.3386192, 17.3203163, -52.0140724, 52.3358765
4: -32.8203049, 23.1237907, -33.4104805, 23.5588036, -56.3791046, 56.5342712
5: -28.3029995, 22.0537243, -28.8249359, 22.4610405, -50.7640343, 50.8786545
6: -26.0419598, 25.7367916, -26.5285645, 26.2043266, -52.2462807, 52.2653503
7: -28.7562294, 27.0796566, -29.2858906, 27.5519409, -56.3081703, 56.3655396
8: -39.8988686, 18.8702698, -40.5856056, 19.2565842, -59.1554451, 59.4558754
9: -25.2603531, 25.4928799, -25.7252083, 25.9656563, -51.2259979, 51.2180862

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0575630, upper bound: 70.0575169
time: 8.64 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0575630, upper bound: 70.0589436
time: 11.03 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -22.9666634, 18.4029770, -27.0142441, 21.5814724, -44.5481339, 45.4172211
1: -20.6015396, 16.4641228, -24.1366119, 19.2739239, -39.8754654, 40.6007347
2: -26.1690369, 16.2238712, -30.6770401, 19.0589104, -45.2279396, 46.9009094
3: -28.1183910, 13.6651421, -32.9414215, 16.1220722, -44.2404633, 46.6065598
4: -26.7754173, 18.7278004, -31.2525978, 21.9689369, -48.7443466, 49.9803925
5: -22.9817123, 17.9532413, -26.8935280, 20.9975719, -43.9792824, 44.8467712
6: -21.0926285, 20.8087349, -24.7457237, 24.4338379, -45.5264664, 45.5544586
7: -23.2710037, 22.2384739, -27.2918167, 25.8500423, -49.1210480, 49.5302887
8: -32.8659706, 14.8362818, -38.1844749, 17.7769413, -50.6429138, 53.0207558
9: -20.4190731, 20.6388950, -23.9562759, 24.2494736, -44.6685486, 44.5951691

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 212

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0479226, upper bound: 70.0478523
time: 9.20 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0476564, upper bound: 70.0476805
time: 8.81 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -25.0478401, 20.0419865, -27.3337326, 21.8320293, -46.8798637, 47.3757172
1: -22.4078598, 17.9118423, -24.4174576, 19.4953842, -41.9032326, 42.3292961
2: -28.4787540, 17.6732006, -31.0343723, 19.2817249, -47.7604675, 48.7075729
3: -30.5774250, 14.9419155, -33.3255005, 16.3128357, -46.8902588, 48.2674179
4: -29.0530243, 20.3837376, -31.6078911, 22.2238693, -51.2768898, 51.9916229
5: -24.9899559, 19.5031166, -27.2034245, 21.2376747, -46.2276306, 46.7065392
6: -22.9575768, 22.6722527, -25.0346107, 24.7194748, -47.6770515, 47.7068558
7: -25.3365803, 24.0735798, -27.6116371, 26.1378498, -51.4744263, 51.6852188
8: -35.5476570, 16.3760262, -38.6063919, 18.0042210, -53.5518684, 54.9824181
9: -22.2291183, 22.4931049, -24.2373161, 24.5325565, -46.7616730, 46.7304230

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 212

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0569756, upper bound: 70.0570838
time: 10.82 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0569756, upper bound: 70.0586096
time: 9.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.45 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 21.45
Output dim: 8, lower bound: -70.0564503, upper bound: 70.0563660
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 21.45
Output dim: 8, lower bound: -70.0566293, upper bound: 70.0566093
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 21.45
Output dim: 8, lower bound: -70.0571666, upper bound: 70.0571855
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 21.45
Output dim: 8, lower bound: -70.0571743, upper bound: 70.0572364
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 21.45
Output dim: 8, lower bound: -70.0563693, upper bound: 70.0562448
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 21.45
Output dim: 8, lower bound: -70.0565469, upper bound: 70.0564702
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 21.45
Output dim: 8, lower bound: -70.0570755, upper bound: 70.0570206
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 21.45
Output dim: 8, lower bound: -70.0570871, upper bound: 70.0570761
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.45
Output dim: 8, lower bound: -70.0483392, upper bound: 70.0481194
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.45
Output dim: 8, lower bound: -70.0482089, upper bound: 70.0480216
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.45
Output dim: 8, lower bound: -70.0575630, upper bound: 70.0575169
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.45
Output dim: 8, lower bound: -70.0575630, upper bound: 70.0589436
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.45
Output dim: 8, lower bound: -70.0479226, upper bound: 70.0478523
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.45
Output dim: 8, lower bound: -70.0476564, upper bound: 70.0476805
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.45
Output dim: 8, lower bound: -70.0569756, upper bound: 70.0570838
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.45
Output dim: 8, lower bound: -70.0569756, upper bound: 70.0586096

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -17.3598099, 13.9947367, -19.2328949, 15.4904356, -32.8502464, 33.2276306
1: -15.6241169, 12.5483837, -17.3248043, 13.8791485, -29.5032578, 29.8731880
2: -19.8432522, 12.3006897, -21.9647560, 13.6110668, -33.4543190, 34.2654457
3: -21.2696056, 10.3032875, -23.5647316, 11.4357061, -32.7053108, 33.8680153
4: -20.3811798, 14.2177811, -22.5324993, 15.7317839, -36.1129608, 36.7502823
5: -17.4431553, 13.6691103, -19.3006325, 15.1308308, -32.5739861, 32.9697418
6: -15.9558058, 15.7904987, -17.6835136, 17.4892311, -33.4450302, 33.4740067
7: -17.5931644, 17.0282288, -19.5103016, 18.8004551, -36.3936195, 36.5385284
8: -25.2853012, 11.0522823, -27.8754711, 12.3362579, -37.6215591, 38.9277496
9: -15.4560852, 15.6586113, -17.1196766, 17.3348083, -32.7908936, 32.7782898

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 80

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0496887, upper bound: 70.0495814
time: 9.82 seconds

## Relational analysis of IS_A1_A1_B1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0494196, upper bound: 70.0493758
time: 11.42 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -16.5482941, 13.3495007, -19.8445625, 15.9701614, -32.5184517, 33.1940613
1: -14.8939838, 11.9706640, -17.8838882, 14.3000278, -29.1940079, 29.8545513
2: -18.9262810, 11.7320118, -22.6737118, 14.0289783, -32.9552612, 34.4057198
3: -20.2735519, 9.8118916, -24.3115845, 11.7738657, -32.0474129, 34.1234741
4: -19.4506683, 13.5543327, -23.2376328, 16.2129784, -35.6636429, 36.7919617
5: -16.6324959, 13.0380306, -19.8955421, 15.5893431, -32.2218361, 32.9335709
6: -15.2009020, 15.0596600, -18.2263260, 18.0362473, -33.2371483, 33.2859840
7: -16.7657814, 16.2660942, -20.1347656, 19.3868904, -36.1526718, 36.4008598
8: -24.1860447, 10.4842291, -28.7446613, 12.7071686, -36.8932114, 39.2288895
9: -14.7327900, 14.9280052, -17.6586761, 17.8673840, -32.6001740, 32.5866814

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 80

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0396553, upper bound: 70.0393548
time: 10.43 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0391089, upper bound: 70.0389684
time: 10.82 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -18.2978497, 14.7379093, -21.5794640, 17.3425713, -35.6404190, 36.3173752
1: -16.4802666, 13.2154312, -19.4446487, 15.5297089, -32.0099754, 32.6600800
2: -20.9220238, 12.9531384, -24.6591492, 15.2396431, -36.1616669, 37.6122894
3: -22.4482021, 10.8622370, -26.4659538, 12.8126373, -35.2608414, 37.3281898
4: -21.4742489, 14.9823570, -25.2532940, 17.6255798, -39.0998306, 40.2356491
5: -18.3875313, 14.3954201, -21.6354065, 16.9229755, -35.3105049, 36.0308228
6: -16.8181896, 16.6370468, -19.8229485, 19.6113319, -36.4295197, 36.4599953
7: -18.5565414, 17.9258766, -21.9109764, 21.0221443, -39.5786858, 39.8368454
8: -26.5930214, 11.6640348, -31.1254807, 13.8583698, -40.4513893, 42.7895088
9: -16.2980881, 16.4982681, -19.2068672, 19.4249153, -35.7230034, 35.7051353

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 175

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0506839, upper bound: 70.0505597
time: 9.84 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0504670, upper bound: 70.0503965
time: 9.57 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -17.4617767, 14.0762844, -22.0856743, 17.7406616, -35.2024384, 36.1619530
1: -15.7329807, 12.6231747, -19.9068222, 15.8749580, -31.6079388, 32.5299950
2: -19.9814472, 12.3687124, -25.2451859, 15.5862513, -35.5676994, 37.6138916
3: -21.4271679, 10.3561573, -27.0837593, 13.0908871, -34.5180511, 37.4399185
4: -20.5220795, 14.3015175, -25.8358803, 18.0186920, -38.5407715, 40.1373978
5: -17.5584183, 13.7470722, -22.1257572, 17.2987347, -34.8571548, 35.8728294
6: -16.0403824, 15.8865128, -20.2702942, 20.0653172, -36.1056976, 36.1568069
7: -17.7068233, 17.1488266, -22.4242496, 21.5075302, -39.2143517, 39.5730743
8: -25.4710484, 11.0740414, -31.8436279, 14.1610737, -39.6321220, 42.9176636
9: -15.5537844, 15.7506018, -19.6509190, 19.8653145, -35.4190979, 35.4015198

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0402333, upper bound: 70.0398883
time: 10.13 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -70.0395585, upper bound: 70.0389684
time: 203.49 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 214.95 seconds
IS_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 214.95
Output dim: 8, lower bound: -70.0496887, upper bound: 70.0495814
IS_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 214.95
Output dim: 8, lower bound: -70.0494196, upper bound: 70.0493758
IS_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 214.95
Output dim: 8, lower bound: -70.0396553, upper bound: 70.0393548
IS_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 214.95
Output dim: 8, lower bound: -70.0391089, upper bound: 70.0389684
IS_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 214.95
Output dim: 8, lower bound: -70.0506839, upper bound: 70.0505597
IS_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 214.95
Output dim: 8, lower bound: -70.0504670, upper bound: 70.0503965
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 214.95
Output dim: 8, lower bound: -70.0402333, upper bound: 70.0398883
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 214.95
Output dim: 8, lower bound: -70.0395585, upper bound: 70.0389684
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 214.95
Output dim: 8, lower bound: -70.0563693, upper bound: 70.0562448
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 214.95
Output dim: 8, lower bound: -70.0565469, upper bound: 70.0564702
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 214.95
Output dim: 8, lower bound: -70.0570755, upper bound: 70.0570206
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 214.95
Output dim: 8, lower bound: -70.0570871, upper bound: 70.0570761
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 214.95
Output dim: 8, lower bound: -70.0483392, upper bound: 70.0481194
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 214.95
Output dim: 8, lower bound: -70.0482089, upper bound: 70.0480216
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 214.95
Output dim: 8, lower bound: -70.0575630, upper bound: 70.0575169
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 214.95
Output dim: 8, lower bound: -70.0575630, upper bound: 70.0589436
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 214.95
Output dim: 8, lower bound: -70.0479226, upper bound: 70.0478523
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 214.95
Output dim: 8, lower bound: -70.0476564, upper bound: 70.0476805
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 214.95
Output dim: 8, lower bound: -70.0569756, upper bound: 70.0570838
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 214.95
Output dim: 8, lower bound: -70.0569756, upper bound: 70.0586096
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=82.72178649902344
rel_dist={8: [-70.08140925617062, 70.08140925618659]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1820.30 seconds
