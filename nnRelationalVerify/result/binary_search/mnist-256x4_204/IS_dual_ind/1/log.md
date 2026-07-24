## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 36.489946449
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-26.4216805, 18.9908962, -26.4216805, 18.9908962, -45.4125748, 45.4125748)
1: (-19.1472130, 19.0538960, -19.1472130, 19.0538960, -38.2011108, 38.2011108)
2: (-26.9148560, 18.2566853, -26.9148560, 18.2566853, -45.1715393, 45.1715393)
3: (-30.9281616, 15.2985239, -30.9281616, 15.2985239, -46.2266846, 46.2266846)
4: (-31.7014198, 18.3034821, -31.7014198, 18.3034821, -50.0048981, 50.0048904)
5: (-27.8828411, 16.3154755, -27.8828411, 16.3154755, -44.1983147, 44.1983109)
6: (-31.4614029, 15.5846272, -31.4614029, 15.5846272, -47.0460281, 47.0460243)
7: (-23.7456284, 23.2433929, -23.7456284, 23.2433929, -46.9890213, 46.9890175)
8: (-34.2775040, 16.8623047, -34.2775040, 16.8623047, -51.1398087, 51.1398087)
9: (-22.6382275, 22.9516029, -22.6382275, 22.9516029, -45.5898285, 45.5898285)

## BASE Result
execution time: IAR + LP analysis = 1.06 + 9.02 = 10.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -36.5032747, upper bound: 36.5032750


# Binary Search by BASE starts (time budget: 2689.91 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=47.04602813720703
rel_dist={6: [-36.503194103113984, 36.50319409863076]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=47.04602813720703
rel_dist={6: [-36.50308478066641, 36.50308478066641]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=47.04602813720703
rel_dist={6: [-36.50297363759564, 36.50297363759566]}

## Binary Search Result
Binary search time: 35.33 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2654.59 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5030224, upper bound: 36.5029282
time: 7.68 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5031944, upper bound: 36.5031941
time: 9.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.10 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.10
Output dim: 6, lower bound: -36.5030224, upper bound: 36.5029282
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.10
Output dim: 6, lower bound: -36.5031944, upper bound: 36.5031941

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -26.1358681, 18.7964439, -26.0028172, 18.6902142, -44.8260803, 44.7992630
1: -18.9409447, 18.8499107, -18.8393440, 18.7579384, -37.6988831, 37.6892509
2: -26.6191959, 18.0804043, -26.4820061, 17.9728203, -44.5920181, 44.5624084
3: -30.5916786, 15.1372766, -30.4374142, 15.0616207, -45.6532974, 45.5746841
4: -31.3408966, 18.1206932, -31.2014637, 18.0157375, -49.3566360, 49.3221588
5: -27.5593796, 16.1499977, -27.4436760, 16.0587063, -43.6180878, 43.5936699
6: -31.0921783, 15.4537849, -30.9776993, 15.3297977, -46.4219742, 46.4314804
7: -23.4900723, 22.9867249, -23.3671112, 22.8802452, -46.3703079, 46.3538361
8: -33.9014206, 16.6999149, -33.7381554, 16.5947380, -50.4961586, 50.4380722
9: -22.3984489, 22.7051811, -22.2801571, 22.5884876, -44.9869385, 44.9853363

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5028640, upper bound: 36.5028640
time: 8.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5028640, upper bound: 36.5029282
time: 6.97 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -25.8575134, 18.5841770, -26.4216805, 18.9908962, -44.8484116, 45.0058594
1: -18.7319984, 18.6548462, -19.1472130, 19.0538960, -37.7858963, 37.8020592
2: -26.3309937, 17.8734856, -26.9148560, 18.2566853, -44.5876770, 44.7883377
3: -30.2666035, 14.9785585, -30.9281616, 15.2985239, -45.5651283, 45.9067192
4: -31.0305214, 17.9139519, -31.7014198, 18.3034821, -49.3340034, 49.6153717
5: -27.2929516, 15.9670887, -27.8828411, 16.3154755, -43.6084213, 43.8499298
6: -30.8138809, 15.2353678, -31.4614029, 15.5846272, -46.3985062, 46.6967697
7: -23.2353802, 22.7546539, -23.7456284, 23.2433929, -46.4787750, 46.5002823
8: -33.5508270, 16.4978981, -34.2775040, 16.8623047, -50.4131317, 50.7754021
9: -22.1546326, 22.4615669, -22.6382275, 22.9516029, -45.1062355, 45.0997925

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5029282, upper bound: 36.5030224
time: 5.23 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5029282, upper bound: 36.5031943
time: 7.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.07 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.07
Output dim: 6, lower bound: -36.5028640, upper bound: 36.5028640
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.07
Output dim: 6, lower bound: -36.5028640, upper bound: 36.5029282
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.07
Output dim: 6, lower bound: -36.5029282, upper bound: 36.5030224
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.07
Output dim: 6, lower bound: -36.5029282, upper bound: 36.5031943

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -26.1358681, 18.7964439, -26.1358681, 18.7964439, -44.9323044, 44.9323044
1: -18.9409447, 18.8499107, -18.9409447, 18.8499107, -37.7908554, 37.7908554
2: -26.6191959, 18.0804043, -26.6191959, 18.0804043, -44.6996002, 44.6996002
3: -30.5916786, 15.1372766, -30.5916786, 15.1372766, -45.7289543, 45.7289543
4: -31.3408966, 18.1206932, -31.3408966, 18.1206932, -49.4615898, 49.4615898
5: -27.5593796, 16.1499977, -27.5593796, 16.1499977, -43.7093773, 43.7093773
6: -31.0921783, 15.4537849, -31.0921783, 15.4537849, -46.5459595, 46.5459595
7: -23.4900723, 22.9867249, -23.4900723, 22.9867249, -46.4767876, 46.4767952
8: -33.9014206, 16.6999149, -33.9014206, 16.6999149, -50.6013336, 50.6013336
9: -22.3984489, 22.7051811, -22.3984489, 22.7051811, -45.1036301, 45.1036301

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4978532, upper bound: 36.4982458
time: 6.94 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4959185, upper bound: 36.4959185
time: 7.52 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -26.1358681, 18.7964439, -25.8575134, 18.5841770, -44.7200432, 44.6539574
1: -18.9409447, 18.8499107, -18.7319984, 18.6548462, -37.5957870, 37.5819092
2: -26.6191959, 18.0804043, -26.3309937, 17.8734856, -44.4926834, 44.4113998
3: -30.5916786, 15.1372766, -30.2666035, 14.9785585, -45.5702362, 45.4038773
4: -31.3408966, 18.1206932, -31.0305214, 17.9139519, -49.2548447, 49.1512146
5: -27.5593796, 16.1499977, -27.2929516, 15.9670887, -43.5264664, 43.4429474
6: -31.0921783, 15.4537849, -30.8138809, 15.2353678, -46.3275452, 46.2676582
7: -23.4900723, 22.9867249, -23.2353802, 22.7546539, -46.2447166, 46.2221069
8: -33.9014206, 16.6999149, -33.5508270, 16.4978981, -50.3993149, 50.2507401
9: -22.3984489, 22.7051811, -22.1546326, 22.4615669, -44.8600159, 44.8598137

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4978532, upper bound: 36.4984493
time: 6.79 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4959185, upper bound: 36.4960450
time: 6.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -25.8575134, 18.5841770, -26.1358681, 18.7964439, -44.6539574, 44.7200470
1: -18.7319984, 18.6548462, -18.9409447, 18.8499107, -37.5819054, 37.5957909
2: -26.3309937, 17.8734856, -26.6191959, 18.0804043, -44.4113998, 44.4926834
3: -30.2666035, 14.9785585, -30.5916786, 15.1372766, -45.4038811, 45.5702362
4: -31.0305214, 17.9139519, -31.3408966, 18.1206932, -49.1512146, 49.2548447
5: -27.2929516, 15.9670887, -27.5593796, 16.1499977, -43.4429474, 43.5264664
6: -30.8138809, 15.2353678, -31.0921783, 15.4537849, -46.2676620, 46.3275452
7: -23.2353802, 22.7546539, -23.4900723, 22.9867249, -46.2221031, 46.2447166
8: -33.5508270, 16.4978981, -33.9014206, 16.6999149, -50.2507401, 50.3993187
9: -22.1546326, 22.4615669, -22.3984489, 22.7051811, -44.8598137, 44.8600159

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4978906, upper bound: 36.4983441
time: 8.47 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4960450, upper bound: 36.4962289
time: 9.89 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -25.8575134, 18.5841770, -25.8575134, 18.5841770, -44.4416885, 44.4416885
1: -18.7319984, 18.6548462, -18.7319984, 18.6548462, -37.3868446, 37.3868446
2: -26.3309937, 17.8734856, -26.3309937, 17.8734856, -44.2044792, 44.2044792
3: -30.2666035, 14.9785585, -30.2666035, 14.9785585, -45.2451630, 45.2451630
4: -31.0305214, 17.9139519, -31.0305214, 17.9139519, -48.9444733, 48.9444733
5: -27.2929516, 15.9670887, -27.2929516, 15.9670887, -43.2600365, 43.2600365
6: -30.8138809, 15.2353678, -30.8138809, 15.2353678, -46.0492477, 46.0492477
7: -23.2353802, 22.7546539, -23.2353802, 22.7546539, -45.9900322, 45.9900360
8: -33.5508270, 16.4978981, -33.5508270, 16.4978981, -50.0487213, 50.0487213
9: -22.1546326, 22.4615669, -22.1546326, 22.4615669, -44.6161995, 44.6161995

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4978906, upper bound: 36.4987367
time: 11.35 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4960450, upper bound: 36.4966518
time: 7.30 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.75 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.75
Output dim: 6, lower bound: -36.4978532, upper bound: 36.4982458
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.75
Output dim: 6, lower bound: -36.4959185, upper bound: 36.4959185
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.75
Output dim: 6, lower bound: -36.4978532, upper bound: 36.4984493
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.75
Output dim: 6, lower bound: -36.4959185, upper bound: 36.4960450
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.75
Output dim: 6, lower bound: -36.4978906, upper bound: 36.4983441
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.75
Output dim: 6, lower bound: -36.4960450, upper bound: 36.4962289
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.75
Output dim: 6, lower bound: -36.4978906, upper bound: 36.4987367
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.75
Output dim: 6, lower bound: -36.4960450, upper bound: 36.4966518

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -25.2156677, 18.1313858, -26.1358681, 18.7964439, -44.0121117, 44.2672501
1: -18.2605801, 18.1958218, -18.9409447, 18.8499107, -37.1104889, 37.1367645
2: -25.6641197, 17.4522133, -26.6191959, 18.0804043, -43.7445221, 44.0714111
3: -29.5050297, 14.6125860, -30.5916786, 15.1372766, -44.6423073, 45.2042618
4: -30.2474709, 17.4760818, -31.3408966, 18.1206932, -48.3681641, 48.8169785
5: -26.6001987, 15.5799351, -27.5593796, 16.1499977, -42.7501945, 43.1393127
6: -30.0336323, 14.8771553, -31.0921783, 15.4537849, -45.4874115, 45.9693336
7: -22.6538429, 22.1870232, -23.4900723, 22.9867249, -45.6405678, 45.6770897
8: -32.7215881, 16.1005764, -33.9014206, 16.6999149, -49.4215012, 50.0019989
9: -21.6044998, 21.9013443, -22.3984489, 22.7051811, -44.3096809, 44.2997894

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4959185, upper bound: 36.4959185
time: 7.97 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4959185, upper bound: 36.4959185
time: 5.85 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -30.7996674, 22.1156044, -25.8613548, 18.5984688, -49.3981361, 47.9769592
1: -22.3406544, 22.1226177, -18.7384319, 18.6551132, -40.9957657, 40.8610497
2: -31.3990288, 21.2742290, -26.3345108, 17.8938522, -49.2928772, 47.6087341
3: -36.0150986, 17.7484093, -30.2681141, 14.9810753, -50.9961700, 48.0165253
4: -36.9039841, 21.2909508, -31.0156021, 17.9285545, -54.8325386, 52.3065529
5: -32.4532852, 18.9799633, -27.2749405, 15.9798670, -48.4331512, 46.2548981
6: -36.4857178, 18.2060566, -30.7794991, 15.2808056, -51.7665215, 48.9855423
7: -27.6695824, 27.0157185, -23.2408695, 22.7495403, -50.4191208, 50.2565804
8: -39.9683838, 19.6333294, -33.5516968, 16.5213909, -56.4897690, 53.1850128
9: -26.3486767, 26.7226791, -22.1614647, 22.4655361, -48.8142090, 48.8841400

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4956259, upper bound: 36.4957153
time: 6.72 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4959185, upper bound: 36.4959185
time: 6.47 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -25.2156677, 18.1313858, -25.8575134, 18.5841770, -43.7998428, 43.9888992
1: -18.2605801, 18.1958218, -18.7319984, 18.6548462, -36.9154243, 36.9278183
2: -25.6641197, 17.4522133, -26.3309937, 17.8734856, -43.5376053, 43.7832069
3: -29.5050297, 14.6125860, -30.2666035, 14.9785585, -44.4835892, 44.8791809
4: -30.2474709, 17.4760818, -31.0305214, 17.9139519, -48.1614227, 48.5066032
5: -26.6001987, 15.5799351, -27.2929516, 15.9670887, -42.5672798, 42.8728790
6: -30.0336323, 14.8771553, -30.8138809, 15.2353678, -45.2690010, 45.6910324
7: -22.6538429, 22.1870232, -23.2353802, 22.7546539, -45.4084969, 45.4224014
8: -32.7215881, 16.1005764, -33.5508270, 16.4978981, -49.2194824, 49.6514053
9: -21.6044998, 21.9013443, -22.1546326, 22.4615669, -44.0660667, 44.0559769

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4962285, upper bound: 36.4960449
time: 4.64 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4962285, upper bound: 36.4960450
time: 6.43 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -30.7996674, 22.1156044, -25.5823078, 18.3862057, -49.1858597, 47.6979141
1: -22.3406544, 22.1226177, -18.5288944, 18.4598198, -40.8004761, 40.6515045
2: -31.3990288, 21.2742290, -26.0451813, 17.6873436, -49.0863686, 47.3194008
3: -36.0150986, 17.7484093, -29.9417343, 14.8213882, -50.8364830, 47.6901436
4: -36.9039841, 21.2909508, -30.7033501, 17.7214012, -54.6253853, 51.9942894
5: -32.4532852, 18.9799633, -27.0062294, 15.7971735, -48.2504578, 45.9861908
6: -36.4857178, 18.2060566, -30.4983978, 15.0638981, -51.5496140, 48.7044411
7: -27.6695824, 27.0157185, -22.9851913, 22.5159569, -50.1855392, 50.0009079
8: -39.9683838, 19.6333294, -33.1984367, 16.3191338, -56.2875175, 52.8317566
9: -26.3486767, 26.7226791, -21.9171600, 22.2208595, -48.5695343, 48.6398392

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4959168, upper bound: 36.4958401
time: 5.39 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4962285, upper bound: 36.4960450
time: 7.46 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -24.9236164, 17.9113770, -26.1358681, 18.7964439, -43.7200623, 44.0472450
1: -18.0417709, 17.9920387, -18.9409447, 18.8499107, -36.8916817, 36.9329796
2: -25.3621922, 17.2395821, -26.6191959, 18.0804043, -43.4425888, 43.8587799
3: -29.1629486, 14.4455013, -30.5916786, 15.1372766, -44.3002243, 45.0371742
4: -29.9174595, 17.2613182, -31.3408966, 18.1206932, -48.0381546, 48.6022110
5: -26.3166542, 15.3909750, -27.5593796, 16.1499977, -42.4666519, 42.9503517
6: -29.7371349, 14.6559563, -31.0921783, 15.4537849, -45.1909103, 45.7481346
7: -22.3875446, 21.9421635, -23.4900723, 22.9867249, -45.3742676, 45.4322319
8: -32.3512535, 15.8911505, -33.9014206, 16.6999149, -49.0511703, 49.7925720
9: -21.3499603, 21.6447296, -22.3984489, 22.7051811, -44.0551376, 44.0431786

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4960450, upper bound: 36.4962289
time: 6.88 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4960450, upper bound: 36.4962285
time: 5.51 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -30.3537922, 21.7891445, -25.8613548, 18.5984688, -48.9522629, 47.6504974
1: -22.0120277, 21.8131371, -18.7384319, 18.6551132, -40.6671371, 40.5515671
2: -30.9267426, 20.9393082, -26.3345108, 17.8938522, -48.8205872, 47.2738113
3: -35.5057335, 17.5038147, -30.2681141, 14.9810753, -50.4868088, 47.7719269
4: -36.3900909, 20.9788685, -31.0156021, 17.9285545, -54.3186455, 51.9944687
5: -32.0114479, 18.7120667, -27.2749405, 15.9798670, -47.9913101, 45.9870071
6: -35.9950752, 17.9053230, -30.7794991, 15.2808056, -51.2758789, 48.6848145
7: -27.2673702, 26.6404285, -23.2408695, 22.7495403, -50.0169106, 49.8812904
8: -39.3970299, 19.3385429, -33.5516968, 16.5213909, -55.9184189, 52.8902359
9: -25.9738903, 26.3407898, -22.1614647, 22.4655361, -48.4394226, 48.5022507

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4957628, upper bound: 36.4960040
time: 9.08 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4960450, upper bound: 36.4962289
time: 9.09 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -24.9236164, 17.9113770, -25.8575134, 18.5841770, -43.5077934, 43.7688904
1: -18.0417709, 17.9920387, -18.7319984, 18.6548462, -36.6966171, 36.7240372
2: -25.3621922, 17.2395821, -26.3309937, 17.8734856, -43.2356796, 43.5705757
3: -29.1629486, 14.4455013, -30.2666035, 14.9785585, -44.1415062, 44.7120972
4: -29.9174595, 17.2613182, -31.0305214, 17.9139519, -47.8314095, 48.2918396
5: -26.3166542, 15.3909750, -27.2929516, 15.9670887, -42.2837448, 42.6839180
6: -29.7371349, 14.6559563, -30.8138809, 15.2353678, -44.9724998, 45.4698372
7: -22.3875446, 21.9421635, -23.2353802, 22.7546539, -45.1421890, 45.1775436
8: -32.3512535, 15.8911505, -33.5508270, 16.4978981, -48.8491516, 49.4419785
9: -21.3499603, 21.6447296, -22.1546326, 22.4615669, -43.8115234, 43.7993622

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4966449, upper bound: 36.4966518
time: 6.94 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4966449, upper bound: 36.4966518
time: 7.10 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -30.3537922, 21.7891445, -25.5823078, 18.3862057, -48.7399902, 47.3714523
1: -22.0120277, 21.8131371, -18.5288944, 18.4598198, -40.4718475, 40.3420334
2: -30.9267426, 20.9393082, -26.0451813, 17.6873436, -48.6140823, 46.9844780
3: -35.5057335, 17.5038147, -29.9417343, 14.8213882, -50.3271217, 47.4455452
4: -36.3900909, 20.9788685, -30.7033501, 17.7214012, -54.1114883, 51.6822014
5: -32.0114479, 18.7120667, -27.0062294, 15.7971735, -47.8086166, 45.7182961
6: -35.9950752, 17.9053230, -30.4983978, 15.0638981, -51.0589752, 48.4037132
7: -27.2673702, 26.6404285, -22.9851913, 22.5159569, -49.7833252, 49.6256180
8: -39.3970299, 19.3385429, -33.1984367, 16.3191338, -55.7161636, 52.5369797
9: -25.9738903, 26.3407898, -21.9171600, 22.2208595, -48.1947479, 48.2579498

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4962934, upper bound: 36.4964066
time: 10.18 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4966449, upper bound: 36.4966518
time: 6.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 17.99 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 6, lower bound: -36.4959185, upper bound: 36.4959185
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 6, lower bound: -36.4959185, upper bound: 36.4959185
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 6, lower bound: -36.4956259, upper bound: 36.4957153
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 6, lower bound: -36.4959185, upper bound: 36.4959185
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 6, lower bound: -36.4962285, upper bound: 36.4960449
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 6, lower bound: -36.4962285, upper bound: 36.4960450
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 6, lower bound: -36.4959168, upper bound: 36.4958401
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 6, lower bound: -36.4962285, upper bound: 36.4960450
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 6, lower bound: -36.4960450, upper bound: 36.4962289
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 6, lower bound: -36.4960450, upper bound: 36.4962285
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 6, lower bound: -36.4957628, upper bound: 36.4960040
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 6, lower bound: -36.4960450, upper bound: 36.4962289
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 6, lower bound: -36.4966449, upper bound: 36.4966518
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 6, lower bound: -36.4966449, upper bound: 36.4966518
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 6, lower bound: -36.4962934, upper bound: 36.4964066
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 6, lower bound: -36.4966449, upper bound: 36.4966518

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -25.2156677, 18.1313858, -25.2156677, 18.1313858, -43.3470535, 43.3470535
1: -18.2605801, 18.1958218, -18.2605801, 18.1958218, -36.4563980, 36.4563980
2: -25.6641197, 17.4522133, -25.6641197, 17.4522133, -43.1163330, 43.1163330
3: -29.5050297, 14.6125860, -29.5050297, 14.6125860, -44.1176109, 44.1176109
4: -30.2474709, 17.4760818, -30.2474709, 17.4760818, -47.7235527, 47.7235527
5: -26.6001987, 15.5799351, -26.6001987, 15.5799351, -42.1801262, 42.1801262
6: -30.0336323, 14.8771553, -30.0336323, 14.8771553, -44.9107895, 44.9107895
7: -22.6538429, 22.1870232, -22.6538429, 22.1870232, -44.8408661, 44.8408661
8: -32.7215881, 16.1005764, -32.7215881, 16.1005764, -48.8221664, 48.8221664
9: -21.6044998, 21.9013443, -21.6044998, 21.9013443, -43.5058441, 43.5058441

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4978299, upper bound: 36.4982436
time: 7.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4978532, upper bound: 36.4982458
time: 12.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -25.2156677, 18.1313858, -30.7996674, 22.1156044, -47.3312721, 48.9310493
1: -18.2605801, 18.1958218, -22.3406544, 22.1226177, -40.3831940, 40.5364723
2: -25.6641197, 17.4522133, -31.3990288, 21.2742290, -46.9383469, 48.8512383
3: -29.5050297, 14.6125860, -36.0150986, 17.7484093, -47.2534409, 50.6276855
4: -30.2474709, 17.4760818, -36.9039841, 21.2909508, -51.5384216, 54.3800659
5: -26.6001987, 15.5799351, -32.4532852, 18.9799633, -45.5801620, 48.0332184
6: -30.0336323, 14.8771553, -36.4857178, 18.2060566, -48.2396812, 51.3628693
7: -22.6538429, 22.1870232, -27.6695824, 27.0157185, -49.6695633, 49.8565979
8: -32.7215881, 16.1005764, -39.9683838, 19.6333294, -52.3549118, 56.0689468
9: -21.6044998, 21.9013443, -26.3486767, 26.7226791, -48.3271790, 48.2500229

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4978299, upper bound: 36.4982436
time: 9.17 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4978532, upper bound: 36.4982458
time: 8.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -30.3856869, 21.8160877, -24.6340256, 17.7197323, -48.1054192, 46.4501114
1: -22.0357323, 21.8294373, -17.8414230, 17.7856846, -39.8214188, 39.6708527
2: -30.9703655, 20.9915199, -25.0681610, 17.0646667, -48.0350342, 46.0596809
3: -35.5295029, 17.5132599, -28.8330383, 14.2845325, -49.8140259, 46.3462982
4: -36.4135895, 21.0038052, -29.5504761, 17.0912876, -53.5048752, 50.5542831
5: -32.0234299, 18.7224541, -25.9879150, 15.2254858, -47.2489090, 44.7103691
6: -36.0125618, 17.9458809, -29.3316326, 14.5470791, -50.5596390, 47.2775116
7: -27.2937660, 26.6574974, -22.1330433, 21.6812840, -48.9750519, 48.7905388
8: -39.4381714, 19.3638783, -31.9661331, 15.7452183, -55.1833878, 51.3300095
9: -25.9929581, 26.3627567, -21.1185532, 21.4052429, -47.3981934, 47.4813004

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4896433, upper bound: 36.4899428
time: 8.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4886415, upper bound: 36.4887966
time: 8.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -30.7996674, 22.1156044, -25.2068214, 18.1234779, -48.9231453, 47.3224258
1: -22.3406544, 22.1226177, -18.2553864, 18.1914387, -40.5320930, 40.3780060
2: -31.3990288, 21.2742290, -25.6559887, 17.4466476, -48.8456726, 46.9302177
3: -36.0150986, 17.7484093, -29.4984779, 14.6082897, -50.6233902, 47.2468872
4: -36.9039841, 21.2909508, -30.2391014, 17.4731331, -54.3771172, 51.5300369
5: -32.4532852, 18.9799633, -26.5923729, 15.5736675, -48.0269547, 45.5723343
6: -36.4857178, 18.2060566, -30.0269508, 14.8684826, -51.3541946, 48.2330055
7: -27.6695824, 27.0157185, -22.6462250, 22.1809521, -49.8505325, 49.6619339
8: -39.9683838, 19.6333294, -32.7106667, 16.0953884, -56.0637589, 52.3439903
9: -26.3486767, 26.7226791, -21.5982742, 21.8954315, -48.2441101, 48.3209534

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4957153, upper bound: 36.4956259
time: 5.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4957153, upper bound: 36.4959185
time: 7.30 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -25.2156677, 18.1313858, -24.9236164, 17.9113770, -43.1270447, 43.0550003
1: -18.2605801, 18.1958218, -18.0417709, 17.9920387, -36.2526169, 36.2375908
2: -25.6641197, 17.4522133, -25.3621922, 17.2395821, -42.9036980, 42.8143997
3: -29.5050297, 14.6125860, -29.1629486, 14.4455013, -43.9505157, 43.7755356
4: -30.2474709, 17.4760818, -29.9174595, 17.2613182, -47.5087891, 47.3935394
5: -26.6001987, 15.5799351, -26.3166542, 15.3909750, -41.9911652, 41.8965912
6: -30.0336323, 14.8771553, -29.7371349, 14.6559563, -44.6895905, 44.6142807
7: -22.6538429, 22.1870232, -22.3875446, 21.9421635, -44.5960083, 44.5745621
8: -32.7215881, 16.1005764, -32.3512535, 15.8911505, -48.6127396, 48.4518280
9: -21.6044998, 21.9013443, -21.3499603, 21.6447296, -43.2492294, 43.2513008

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4982518, upper bound: 36.4984449
time: 21.91 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4982764, upper bound: 36.4984493
time: 18.28 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -25.2156677, 18.1313858, -30.3537922, 21.7891445, -47.0048141, 48.4851761
1: -18.2605801, 18.1958218, -22.0120277, 21.8131371, -40.0737076, 40.2078476
2: -25.6641197, 17.4522133, -30.9267426, 20.9393082, -46.6034279, 48.3789558
3: -29.5050297, 14.6125860, -35.5057335, 17.5038147, -47.0088348, 50.1183205
4: -30.2474709, 17.4760818, -36.3900909, 20.9788685, -51.2263298, 53.8661613
5: -26.6001987, 15.5799351, -32.0114479, 18.7120667, -45.3122635, 47.5913773
6: -30.0336323, 14.8771553, -35.9950752, 17.9053230, -47.9389458, 50.8722305
7: -22.6538429, 22.1870232, -27.2673702, 26.6404285, -49.2942657, 49.4543877
8: -32.7215881, 16.1005764, -39.3970299, 19.3385429, -52.0601311, 55.4976044
9: -21.6044998, 21.9013443, -25.9738903, 26.3407898, -47.9452896, 47.8752327

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4982518, upper bound: 36.4984449
time: 7.15 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4982764, upper bound: 36.4984493
time: 6.22 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -30.3856869, 21.8160877, -24.3600082, 17.5101395, -47.8958282, 46.1760864
1: -22.0357323, 21.8294373, -17.6338882, 17.5938473, -39.6295776, 39.4633255
2: -30.9703655, 20.9915199, -24.7837639, 16.8594017, -47.8297653, 45.7752838
3: -35.5295029, 17.5132599, -28.5134354, 14.1261196, -49.6556244, 46.0266953
4: -36.4135895, 21.0038052, -29.2487831, 16.8838463, -53.2974319, 50.2525864
5: -32.0234299, 18.7224541, -25.7262974, 15.0429344, -47.0663643, 44.4487534
6: -36.0125618, 17.9458809, -29.0607433, 14.3251200, -50.3376808, 47.0066223
7: -27.2937660, 26.6574974, -21.8813782, 21.4522552, -48.7460136, 48.5388718
8: -39.4381714, 19.3638783, -31.6211700, 15.5412407, -54.9794121, 50.9850426
9: -25.9929581, 26.3627567, -20.8781624, 21.1625767, -47.1555290, 47.2409210

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4899772, upper bound: 36.4901835
time: 8.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4889373, upper bound: 36.4890245
time: 24.97 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -30.7996674, 22.1156044, -24.8784485, 17.8777847, -48.6774521, 46.9940529
1: -22.3406544, 22.1226177, -18.0099678, 17.9624748, -40.3031311, 40.1325836
2: -31.3990288, 21.2742290, -25.3179359, 17.2096596, -48.6086769, 46.5921631
3: -36.0150986, 17.7484093, -29.1149845, 14.4207840, -50.4358826, 46.8633957
4: -36.9039841, 21.2909508, -29.8669434, 17.2335148, -54.1374969, 51.1578903
5: -32.4532852, 18.9799633, -26.2708588, 15.3629541, -47.8162384, 45.2508163
6: -36.4857178, 18.2060566, -29.6875210, 14.6267462, -51.1124611, 47.8935699
7: -27.6695824, 27.0157185, -22.3475914, 21.9049225, -49.5745049, 49.3633118
8: -39.9683838, 19.6333294, -32.2945862, 15.8631516, -55.8315277, 51.9279099
9: -26.3486767, 26.7226791, -21.3135242, 21.6075516, -47.9562302, 48.0362015

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4960039, upper bound: 36.4957628
time: 6.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4960039, upper bound: 36.4960449
time: 7.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -24.9236164, 17.9113770, -25.2156677, 18.1313858, -43.0550003, 43.1270447
1: -18.0417709, 17.9920387, -18.2605801, 18.1958218, -36.2375908, 36.2526093
2: -25.3621922, 17.2395821, -25.6641197, 17.4522133, -42.8144035, 42.9037018
3: -29.1629486, 14.4455013, -29.5050297, 14.6125860, -43.7755356, 43.9505196
4: -29.9174595, 17.2613182, -30.2474709, 17.4760818, -47.3935394, 47.5087891
5: -26.3166542, 15.3909750, -26.6001987, 15.5799351, -41.8965912, 41.9911652
6: -29.7371349, 14.6559563, -30.0336323, 14.8771553, -44.6142807, 44.6895905
7: -22.3875446, 21.9421635, -22.6538429, 22.1870232, -44.5745621, 44.5960083
8: -32.3512535, 15.8911505, -32.7215881, 16.1005764, -48.4518280, 48.6127396
9: -21.3499603, 21.6447296, -21.6044998, 21.9013443, -43.2513046, 43.2492294

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4978675, upper bound: 36.4983433
time: 8.45 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4978906, upper bound: 36.4983441
time: 24.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -24.9236164, 17.9113770, -30.7996674, 22.1156044, -47.0392227, 48.7110443
1: -18.0417709, 17.9920387, -22.3406544, 22.1226177, -40.1643867, 40.3326912
2: -25.3621922, 17.2395821, -31.3990288, 21.2742290, -46.6364136, 48.6386108
3: -29.1629486, 14.4455013, -36.0150986, 17.7484093, -46.9113579, 50.4605942
4: -29.9174595, 17.2613182, -36.9039841, 21.2909508, -51.2083969, 54.1652985
5: -26.3166542, 15.3909750, -32.4532852, 18.9799633, -45.2966156, 47.8442574
6: -29.7371349, 14.6559563, -36.4857178, 18.2060566, -47.9431801, 51.1416740
7: -22.3875446, 21.9421635, -27.6695824, 27.0157185, -49.4032631, 49.6117477
8: -32.3512535, 15.8911505, -39.9683838, 19.6333294, -51.9845810, 55.8595276
9: -21.3499603, 21.6447296, -26.3486767, 26.7226791, -48.0726395, 47.9934082

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4978675, upper bound: 36.4983433
time: 7.20 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4978906, upper bound: 36.4983441
time: 10.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -29.9764118, 21.5158958, -24.6340256, 17.7197323, -47.6961441, 46.1499214
1: -21.7338085, 21.5462055, -17.8414230, 17.7856846, -39.5194893, 39.3876228
2: -30.5361061, 20.6825733, -25.0681610, 17.0646667, -47.6007729, 45.7507324
3: -35.0628815, 17.2888756, -28.8330383, 14.2845325, -49.3473969, 46.1219139
4: -35.9439774, 20.7163162, -29.5504761, 17.0912876, -53.0352631, 50.2667923
5: -31.6193962, 18.4764347, -25.9879150, 15.2254858, -46.8448792, 44.4643478
6: -35.5642471, 17.6665764, -29.3316326, 14.5470791, -50.1113243, 46.9982071
7: -26.9243584, 26.3137245, -22.1330433, 21.6812840, -48.6056442, 48.4467697
8: -38.9140396, 19.0915680, -31.9661331, 15.7452183, -54.6592560, 51.0577011
9: -25.6492882, 26.0122948, -21.1185532, 21.4052429, -47.0545197, 47.1308365

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4897463, upper bound: 36.4901627
time: 8.20 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4887651, upper bound: 36.4890063
time: 7.92 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -30.3537922, 21.7891445, -25.2068214, 18.1234779, -48.4772720, 46.9959641
1: -22.0120277, 21.8131371, -18.2553864, 18.1914387, -40.2034683, 40.0685234
2: -30.9267426, 20.9393082, -25.6559887, 17.4466476, -48.3733902, 46.5952911
3: -35.5057335, 17.5038147, -29.4984779, 14.6082897, -50.1140213, 47.0022926
4: -36.3900909, 20.9788685, -30.2391014, 17.4731331, -53.8632202, 51.2179489
5: -32.0114479, 18.7120667, -26.5923729, 15.5736675, -47.5851135, 45.3044395
6: -35.9950752, 17.9053230, -30.0269508, 14.8684826, -50.8635559, 47.9322739
7: -27.2673702, 26.6404285, -22.6462250, 22.1809521, -49.4483223, 49.2866440
8: -39.3970299, 19.3385429, -32.7106667, 16.0953884, -55.4924088, 52.0492096
9: -25.9738903, 26.3407898, -21.5982742, 21.8954315, -47.8693237, 47.9390640

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4958401, upper bound: 36.4959168
time: 5.34 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4958401, upper bound: 36.4962285
time: 8.06 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -24.9236164, 17.9113770, -24.9236164, 17.9113770, -42.8349915, 42.8349915
1: -18.0417709, 17.9920387, -18.0417709, 17.9920387, -36.0338097, 36.0338058
2: -25.3621922, 17.2395821, -25.3621922, 17.2395821, -42.6017723, 42.6017647
3: -29.1629486, 14.4455013, -29.1629486, 14.4455013, -43.6084518, 43.6084518
4: -29.9174595, 17.2613182, -29.9174595, 17.2613182, -47.1787796, 47.1787796
5: -26.3166542, 15.3909750, -26.3166542, 15.3909750, -41.7076302, 41.7076263
6: -29.7371349, 14.6559563, -29.7371349, 14.6559563, -44.3930893, 44.3930893
7: -22.3875446, 21.9421635, -22.3875446, 21.9421635, -44.3297043, 44.3297081
8: -32.3512535, 15.8911505, -32.3512535, 15.8911505, -48.2424049, 48.2424049
9: -21.3499603, 21.6447296, -21.3499603, 21.6447296, -42.9946899, 42.9946899

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4984201, upper bound: 36.4987308
time: 7.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4984385, upper bound: 36.4987367
time: 13.22 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -24.9236164, 17.9113770, -30.3537922, 21.7891445, -46.7127609, 48.2651672
1: -18.0417709, 17.9920387, -22.0120277, 21.8131371, -39.8549042, 40.0040665
2: -25.3621922, 17.2395821, -30.9267426, 20.9393082, -46.3014984, 48.1663208
3: -29.1629486, 14.4455013, -35.5057335, 17.5038147, -46.6667633, 49.9512291
4: -29.9174595, 17.2613182, -36.3900909, 20.9788685, -50.8963089, 53.6514091
5: -26.3166542, 15.3909750, -32.0114479, 18.7120667, -45.0287209, 47.4024162
6: -29.7371349, 14.6559563, -35.9950752, 17.9053230, -47.6424522, 50.6510315
7: -22.3875446, 21.9421635, -27.2673702, 26.6404285, -49.0279694, 49.2095337
8: -32.3512535, 15.8911505, -39.3970299, 19.3385429, -51.6897964, 55.2881699
9: -21.3499603, 21.6447296, -25.9738903, 26.3407898, -47.6907501, 47.6186218

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4984201, upper bound: 36.4987308
time: 8.49 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4984385, upper bound: 36.4987367
time: 7.23 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -29.9764118, 21.5158958, -24.3600082, 17.5101395, -47.4865494, 45.8758926
1: -21.7338085, 21.5462055, -17.6338882, 17.5938473, -39.3276482, 39.1800919
2: -30.5361061, 20.6825733, -24.7837639, 16.8594017, -47.3955002, 45.4663353
3: -35.0628815, 17.2888756, -28.5134354, 14.1261196, -49.1889877, 45.8023109
4: -35.9439774, 20.7163162, -29.2487831, 16.8838463, -52.8278198, 49.9650993
5: -31.6193962, 18.4764347, -25.7262974, 15.0429344, -46.6623306, 44.2027321
6: -35.5642471, 17.6665764, -29.0607433, 14.3251200, -49.8893623, 46.7273178
7: -26.9243584, 26.3137245, -21.8813782, 21.4522552, -48.3766136, 48.1951027
8: -38.9140396, 19.0915680, -31.6211700, 15.5412407, -54.4552765, 50.7127380
9: -25.6492882, 26.0122948, -20.8781624, 21.1625767, -46.8118629, 46.8904572

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4902903, upper bound: 36.4906581
time: 9.20 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4892423, upper bound: 36.4894309
time: 15.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -30.3537922, 21.7891445, -24.8784485, 17.8777847, -48.2315750, 46.6675949
1: -22.0120277, 21.8131371, -18.0099678, 17.9624748, -39.9745026, 39.8231010
2: -30.9267426, 20.9393082, -25.3179359, 17.2096596, -48.1363945, 46.2572441
3: -35.5057335, 17.5038147, -29.1149845, 14.4207840, -49.9265175, 46.6187973
4: -36.3900909, 20.9788685, -29.8669434, 17.2335148, -53.6236038, 50.8458099
5: -32.0114479, 18.7120667, -26.2708588, 15.3629541, -47.3744011, 44.9829254
6: -35.9950752, 17.9053230, -29.6875210, 14.6267462, -50.6218224, 47.5928345
7: -27.2673702, 26.6404285, -22.3475914, 21.9049225, -49.1722946, 48.9880142
8: -39.3970299, 19.3385429, -32.2945862, 15.8631516, -55.2601776, 51.6331291
9: -25.9738903, 26.3407898, -21.3135242, 21.6075516, -47.5814438, 47.6543121

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4963870, upper bound: 36.4962967
time: 15.18 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4963870, upper bound: 36.4966518
time: 29.09 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 45.43 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4978299, upper bound: 36.4982436
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4978532, upper bound: 36.4982458
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4978299, upper bound: 36.4982436
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4978532, upper bound: 36.4982458
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4896433, upper bound: 36.4899428
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4886415, upper bound: 36.4887966
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4957153, upper bound: 36.4956259
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4957153, upper bound: 36.4959185
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4982518, upper bound: 36.4984449
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4982764, upper bound: 36.4984493
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4982518, upper bound: 36.4984449
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4982764, upper bound: 36.4984493
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4899772, upper bound: 36.4901835
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4889373, upper bound: 36.4890245
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4960039, upper bound: 36.4957628
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4960039, upper bound: 36.4960449
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4978675, upper bound: 36.4983433
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4978906, upper bound: 36.4983441
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4978675, upper bound: 36.4983433
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4978906, upper bound: 36.4983441
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4897463, upper bound: 36.4901627
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4887651, upper bound: 36.4890063
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4958401, upper bound: 36.4959168
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4958401, upper bound: 36.4962285
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4984201, upper bound: 36.4987308
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4984385, upper bound: 36.4987367
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4984201, upper bound: 36.4987308
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4984385, upper bound: 36.4987367
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4902903, upper bound: 36.4906581
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4892423, upper bound: 36.4894309
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4963870, upper bound: 36.4962967
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 45.43
Output dim: 6, lower bound: -36.4963870, upper bound: 36.4966518
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=47.04602813720703
rel_dist={6: [-36.503194103113984, 36.50319409863076]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5028621, upper bound: 36.5028090
time: 7.39 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5030845, upper bound: 36.5030848
time: 11.05 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.54 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 18.54
Output dim: 6, lower bound: -36.5028621, upper bound: 36.5028090
IS_A2, status: Status.UNKNOWN, split count: 1, time: 18.54
Output dim: 6, lower bound: -36.5030845, upper bound: 36.5030848

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -26.1358681, 18.7964439, -25.3455048, 18.2198257, -44.3556862, 44.1419449
1: -18.9409447, 18.8499107, -18.3567581, 18.2943649, -37.2353096, 37.2066689
2: -26.6191959, 18.0804043, -25.8039665, 17.5289192, -44.1481171, 43.8843651
3: -30.5916786, 15.1372766, -29.6666183, 14.6896248, -45.2813034, 44.8038940
4: -31.3408966, 18.1206932, -30.4157219, 17.5645866, -48.9054794, 48.5364151
5: -27.5593796, 16.1499977, -26.7531776, 15.6567087, -43.2160835, 42.9031754
6: -31.0921783, 15.4537849, -30.2165813, 14.9332542, -46.0254326, 45.6703644
7: -23.4900723, 22.9867249, -22.7738781, 22.3097210, -45.7997894, 45.7605972
8: -33.9014206, 16.6999149, -32.8905869, 16.1761780, -50.0775986, 49.5904961
9: -22.3984489, 22.7051811, -21.7187080, 22.0182171, -44.4166641, 44.4238853

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4971996, upper bound: 36.4969147
time: 15.39 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4959845, upper bound: 36.4958717
time: 7.07 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -25.8575134, 18.5841770, -26.3831520, 18.9630966, -44.8206100, 44.9673309
1: -18.7319984, 18.6548462, -19.1188431, 19.0266380, -37.7586365, 37.7736893
2: -26.3309937, 17.8734856, -26.8749542, 18.2304878, -44.5614815, 44.7484398
3: -30.2666035, 14.9785585, -30.8830051, 15.2766914, -45.5432892, 45.8615646
4: -31.0305214, 17.9139519, -31.6556320, 18.2768784, -49.3073997, 49.5695839
5: -27.2929516, 15.9670887, -27.8425751, 16.2916813, -43.5846252, 43.8096619
6: -30.8138809, 15.2353678, -31.4172173, 15.5607271, -46.3746071, 46.6525841
7: -23.2353802, 22.7546539, -23.7107735, 23.2100353, -46.4454155, 46.4654274
8: -33.5508270, 16.4978981, -34.2279358, 16.8373947, -50.3882217, 50.7258339
9: -22.1546326, 22.4615669, -22.6052055, 22.9181633, -45.0727921, 45.0667725

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5028090, upper bound: 36.5028622
time: 8.78 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5028090, upper bound: 36.5030845
time: 23.38 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 33.25 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 33.25
Output dim: 6, lower bound: -36.4971996, upper bound: 36.4969147
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 33.25
Output dim: 6, lower bound: -36.4959845, upper bound: 36.4958717
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 33.25
Output dim: 6, lower bound: -36.5028090, upper bound: 36.5028622
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 33.25
Output dim: 6, lower bound: -36.5028090, upper bound: 36.5030845

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -25.9787121, 18.6829224, -24.4243526, 17.5554790, -43.5341911, 43.1072769
1: -18.8249454, 18.7382221, -17.6749649, 17.6396065, -36.4645462, 36.4131851
2: -26.4562950, 17.9732628, -24.8483963, 16.9035225, -43.3598175, 42.8216591
3: -30.4064198, 15.0479298, -28.5763607, 14.1634846, -44.5699043, 43.6242905
4: -31.1544151, 18.0108833, -29.3165874, 16.9192524, -48.0736656, 47.3274689
5: -27.3962421, 16.0524864, -25.7881241, 15.0886068, -42.4848480, 41.8406105
6: -30.9126358, 15.3547249, -29.1497021, 14.3629742, -45.2756119, 44.5044250
7: -23.3474064, 22.8506889, -21.9376965, 21.5068016, -44.8542099, 44.7883835
8: -33.7008133, 16.5976200, -31.7055702, 15.5780840, -49.2788963, 48.3031921
9: -22.2628441, 22.5681019, -20.9244671, 21.2119560, -43.4748001, 43.4925652

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4968498, upper bound: 36.4964876
time: 9.28 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4971996, upper bound: 36.4969147
time: 8.89 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -25.4580479, 18.3076820, -29.7815781, 21.3779087, -46.8359566, 48.0892563
1: -18.4404316, 18.3690376, -21.6005173, 21.4137306, -39.8541641, 39.9695549
2: -25.9155312, 17.6194077, -30.3298988, 20.5586281, -46.4741516, 47.9493065
3: -29.7918186, 14.7507734, -34.8474922, 17.1847439, -46.9765625, 49.5982590
4: -30.5369358, 17.6455135, -35.7157326, 20.5951519, -51.1320877, 53.3612442
5: -26.8550301, 15.7304630, -31.4243813, 18.3525105, -45.2075424, 47.1548462
6: -30.3164062, 15.0286522, -35.3731842, 17.5370750, -47.8534813, 50.4018364
7: -22.8741512, 22.3994389, -26.7414417, 26.1523533, -49.0265045, 49.1408806
8: -33.0351257, 16.2590714, -38.6725616, 18.9707928, -52.0059052, 54.9316330
9: -21.8134041, 22.1128922, -25.4782791, 25.8540897, -47.6674881, 47.5911713

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4957118, upper bound: 36.4955682
time: 6.35 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4959845, upper bound: 36.4958717
time: 10.47 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -25.8575134, 18.5841770, -26.1358681, 18.7964439, -44.6539574, 44.7200470
1: -18.7319984, 18.6548462, -18.9409447, 18.8499107, -37.5819054, 37.5957909
2: -26.3309937, 17.8734856, -26.6191959, 18.0804043, -44.4113998, 44.4926834
3: -30.2666035, 14.9785585, -30.5916786, 15.1372766, -45.4038811, 45.5702362
4: -31.0305214, 17.9139519, -31.3408966, 18.1206932, -49.1512146, 49.2548447
5: -27.2929516, 15.9670887, -27.5593796, 16.1499977, -43.4429474, 43.5264664
6: -30.8138809, 15.2353678, -31.0921783, 15.4537849, -46.2676620, 46.3275452
7: -23.2353802, 22.7546539, -23.4900723, 22.9867249, -46.2221031, 46.2447166
8: -33.5508270, 16.4978981, -33.9014206, 16.6999149, -50.2507401, 50.3993187
9: -22.1546326, 22.4615669, -22.3984489, 22.7051811, -44.8598137, 44.8600159

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4969147, upper bound: 36.4971996
time: 8.22 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4958717, upper bound: 36.4959845
time: 8.22 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -25.8575134, 18.5841770, -25.8575134, 18.5841770, -44.4416885, 44.4416885
1: -18.7319984, 18.6548462, -18.7319984, 18.6548462, -37.3868446, 37.3868446
2: -26.3309937, 17.8734856, -26.3309937, 17.8734856, -44.2044792, 44.2044792
3: -30.2666035, 14.9785585, -30.2666035, 14.9785585, -45.2451630, 45.2451630
4: -31.0305214, 17.9139519, -31.0305214, 17.9139519, -48.9444733, 48.9444733
5: -27.2929516, 15.9670887, -27.2929516, 15.9670887, -43.2600365, 43.2600365
6: -30.8138809, 15.2353678, -30.8138809, 15.2353678, -46.0492477, 46.0492477
7: -23.2353802, 22.7546539, -23.2353802, 22.7546539, -45.9900322, 45.9900360
8: -33.5508270, 16.4978981, -33.5508270, 16.4978981, -50.0487213, 50.0487213
9: -22.1546326, 22.4615669, -22.1546326, 22.4615669, -44.6161995, 44.6161995

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4969147, upper bound: 36.4977601
time: 8.55 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4958717, upper bound: 36.4965346
time: 28.86 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 38.52 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 38.52
Output dim: 6, lower bound: -36.4968498, upper bound: 36.4964876
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 38.52
Output dim: 6, lower bound: -36.4971996, upper bound: 36.4969147
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 38.52
Output dim: 6, lower bound: -36.4957118, upper bound: 36.4955682
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 38.52
Output dim: 6, lower bound: -36.4959845, upper bound: 36.4958717
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 38.52
Output dim: 6, lower bound: -36.4969147, upper bound: 36.4971996
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 38.52
Output dim: 6, lower bound: -36.4958717, upper bound: 36.4959845
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 38.52
Output dim: 6, lower bound: -36.4969147, upper bound: 36.4977601
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 38.52
Output dim: 6, lower bound: -36.4958717, upper bound: 36.4965346

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -24.7387066, 17.7954140, -23.4529305, 16.8517704, -41.5904770, 41.2483368
1: -17.9190502, 17.8599777, -16.9564667, 16.9496613, -34.8687134, 34.8164444
2: -25.1771355, 17.1360683, -23.8436546, 16.2405739, -41.4177017, 40.9797211
3: -28.9569740, 14.3445644, -27.4332027, 13.6091690, -42.5661392, 41.7777634
4: -29.6746178, 17.1653023, -28.1627235, 16.2416134, -45.9162292, 45.3280258
5: -26.0969582, 15.2902641, -24.7701645, 14.4864120, -40.5833702, 40.0604248
6: -29.4518929, 14.6127911, -28.0196972, 13.7591534, -43.2110405, 42.6324844
7: -22.2283058, 21.7722740, -21.0561428, 20.6616096, -42.8899078, 42.8284149
8: -32.1001816, 15.8135033, -30.4537716, 14.9478836, -47.0480652, 46.2672729
9: -21.2091331, 21.4971275, -20.0904427, 20.3638535, -41.5729866, 41.5875702

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4798694, upper bound: 36.4792598
time: 9.15 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4783684, upper bound: 36.4778897
time: 9.40 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -25.3202496, 18.2053394, -24.3250599, 17.4836311, -42.8038788, 42.5303993
1: -18.3395081, 18.2720432, -17.6015625, 17.5691681, -35.9086761, 35.8736038
2: -25.7739754, 17.5239296, -24.7457542, 16.8359222, -42.6098976, 42.2696724
3: -29.6329422, 14.6733255, -28.4594860, 14.1069241, -43.7398682, 43.1328125
4: -30.3738632, 17.5533257, -29.1984005, 16.8500957, -47.2239609, 46.7517242
5: -26.7107792, 15.6437016, -25.6839695, 15.0273647, -41.7381439, 41.3276672
6: -30.1578693, 14.9391203, -29.0343571, 14.3016300, -44.4594994, 43.9734726
7: -22.7493706, 22.2796650, -21.8476791, 21.4202881, -44.1696587, 44.1273346
8: -32.8561821, 16.1690884, -31.5775986, 15.5139341, -48.3701096, 47.7466850
9: -21.6963978, 21.9950066, -20.8392258, 21.1253166, -42.8217087, 42.8342285

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4971345, upper bound: 36.4968439
time: 9.13 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4971345, upper bound: 36.4969147
time: 12.55 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -24.2528858, 17.4435883, -28.8437443, 20.6986389, -44.9515228, 46.2873306
1: -17.5585270, 17.5143013, -20.9075947, 20.7495670, -38.3080940, 38.4218979
2: -24.6718369, 16.8036156, -29.3605175, 19.9187164, -44.5905533, 46.1641235
3: -28.3814125, 14.0661440, -33.7453308, 16.6496277, -45.0310402, 47.8114738
4: -29.0973167, 16.8221760, -34.6058044, 19.9413681, -49.0386772, 51.4279709
5: -25.5894642, 14.9893951, -30.4485588, 17.7682343, -43.3576927, 45.4379539
6: -28.8907185, 14.3086224, -34.2963715, 16.9470558, -45.8377762, 48.6049805
7: -21.7863007, 21.3489895, -25.8904915, 25.3388786, -47.1251717, 47.2394791
8: -31.4761429, 15.4964418, -37.4705200, 18.3573017, -49.8334427, 52.9669609
9: -20.7890224, 21.0709343, -24.6728077, 25.0361290, -45.8251495, 45.7437363

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4890172, upper bound: 36.4888183
time: 9.01 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4884242, upper bound: 36.4882911
time: 6.13 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -24.8111839, 17.8372879, -29.6902580, 21.3117962, -46.1229706, 47.5275459
1: -17.9620838, 17.9098415, -21.5330601, 21.3491516, -39.3112335, 39.4429016
2: -25.2448921, 17.1761055, -30.2355289, 20.4965038, -45.7413940, 47.4116364
3: -29.0296211, 14.3817596, -34.7401047, 17.1327019, -46.1623192, 49.1218643
4: -29.7682209, 17.1942215, -35.6075554, 20.5315475, -50.2997665, 52.8017769
5: -26.1785583, 15.3290939, -31.3293076, 18.2957935, -44.4743500, 46.6584015
6: -29.5686817, 14.6220589, -35.2685852, 17.4796753, -47.0483551, 49.8906364
7: -22.2865772, 21.8359604, -26.6585712, 26.0731926, -48.3597717, 48.4945297
8: -32.2019958, 15.8379002, -38.5555573, 18.9111748, -51.1131516, 54.3934555
9: -21.2563820, 21.5486012, -25.3998432, 25.7744045, -47.0307846, 46.9484444

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4956673, upper bound: 36.4956149
time: 8.55 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4956673, upper bound: 36.4958717
time: 7.00 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -24.9236164, 17.9113770, -25.9787121, 18.6829224, -43.6065369, 43.8900909
1: -18.0417709, 17.9920387, -18.8249454, 18.7382221, -36.7799911, 36.8169861
2: -25.3621922, 17.2395821, -26.4562950, 17.9732628, -43.3354568, 43.6958771
3: -29.1629486, 14.4455013, -30.4064198, 15.0479298, -44.2108765, 44.8519211
4: -29.9174595, 17.2613182, -31.1544151, 18.0108833, -47.9283409, 48.4157333
5: -26.3166542, 15.3909750, -27.3962421, 16.0524864, -42.3691406, 42.7872124
6: -29.7371349, 14.6559563, -30.9126358, 15.3547249, -45.0918579, 45.5685921
7: -22.3875446, 21.9421635, -23.3474064, 22.8506889, -45.2382317, 45.2895699
8: -32.3512535, 15.8911505, -33.7008133, 16.5976200, -48.9488678, 49.5919609
9: -21.3499603, 21.6447296, -22.2628441, 22.5681019, -43.9180603, 43.9075737

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4964876, upper bound: 36.4968493
time: 11.49 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4969147, upper bound: 36.4971996
time: 9.42 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -30.3537922, 21.7891445, -25.4580479, 18.3076820, -48.6614647, 47.2471924
1: -22.0120277, 21.8131371, -18.4404316, 18.3690376, -40.3810654, 40.2535706
2: -30.9267426, 20.9393082, -25.9155312, 17.6194077, -48.5461502, 46.8548355
3: -35.5057335, 17.5038147, -29.7918186, 14.7507734, -50.2565002, 47.2956314
4: -36.3900909, 20.9788685, -30.5369358, 17.6455135, -54.0356064, 51.5157928
5: -32.0114479, 18.7120667, -26.8550301, 15.7304630, -47.7419128, 45.5670967
6: -35.9950752, 17.9053230, -30.3164062, 15.0286522, -51.0237274, 48.2217216
7: -27.2673702, 26.6404285, -22.8741512, 22.3994389, -49.6668091, 49.5145798
8: -39.3970299, 19.3385429, -33.0351257, 16.2590714, -55.6561012, 52.3736649
9: -25.9738903, 26.3407898, -21.8134041, 22.1128922, -48.0867844, 48.1541939

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4955682, upper bound: 36.4957118
time: 7.36 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4958715, upper bound: 36.4959845
time: 8.53 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -24.9236164, 17.9113770, -25.7004700, 18.4710350, -43.3946533, 43.6118469
1: -18.0417709, 17.9920387, -18.6160431, 18.5434303, -36.5852013, 36.6080818
2: -25.3621922, 17.2395821, -26.1679764, 17.7669716, -43.1291656, 43.4075546
3: -29.1629486, 14.4455013, -30.0812321, 14.8889790, -44.0519257, 44.5267258
4: -29.9174595, 17.2613182, -30.8435822, 17.8042679, -47.7217255, 48.1048965
5: -26.3166542, 15.3909750, -27.1290512, 15.8700190, -42.1866722, 42.5200233
6: -29.7371349, 14.6559563, -30.6335220, 15.1374550, -44.8745880, 45.2894783
7: -22.3875446, 21.9421635, -23.0926495, 22.6182079, -45.0057526, 45.0348129
8: -32.3512535, 15.8911505, -33.3493919, 16.3957615, -48.7470169, 49.2405396
9: -21.3499603, 21.6447296, -22.0192070, 22.3243504, -43.6743050, 43.6639366

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4970548, upper bound: 36.4973549
time: 8.50 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4975467, upper bound: 36.4977601
time: 10.36 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -30.3537922, 21.7891445, -25.1738510, 18.0922127, -48.4460068, 46.9629974
1: -22.0120277, 21.8131371, -18.2269630, 18.1702213, -40.1822433, 40.0401001
2: -30.9267426, 20.9393082, -25.6212692, 17.4106617, -48.3374023, 46.5605736
3: -35.5057335, 17.5038147, -29.4589024, 14.5879078, -50.0936356, 46.9627151
4: -36.3900909, 20.9788685, -30.2171402, 17.4353600, -53.8254395, 51.1959915
5: -32.0114479, 18.7120667, -26.5799408, 15.5454445, -47.5568848, 45.2920074
6: -35.9950752, 17.9053230, -30.0280666, 14.8106012, -50.8056755, 47.9333878
7: -27.2673702, 26.6404285, -22.6143875, 22.1609840, -49.4283524, 49.2548141
8: -39.3970299, 19.3385429, -32.6745911, 16.0539093, -55.4509354, 52.0131340
9: -25.9738903, 26.3407898, -21.5650024, 21.8631630, -47.8370514, 47.9057922

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4961430, upper bound: 36.4962081
time: 13.21 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4965300, upper bound: 36.4965346
time: 7.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.04 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 22.04
Output dim: 6, lower bound: -36.4798694, upper bound: 36.4792598
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 22.04
Output dim: 6, lower bound: -36.4783684, upper bound: 36.4778897
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 6, lower bound: -36.4971345, upper bound: 36.4968439
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 6, lower bound: -36.4971345, upper bound: 36.4969147
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 22.04
Output dim: 6, lower bound: -36.4890172, upper bound: 36.4888183
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 22.04
Output dim: 6, lower bound: -36.4884242, upper bound: 36.4882911
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 6, lower bound: -36.4956673, upper bound: 36.4956149
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 6, lower bound: -36.4956673, upper bound: 36.4958717
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 6, lower bound: -36.4964876, upper bound: 36.4968493
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 6, lower bound: -36.4969147, upper bound: 36.4971996
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 6, lower bound: -36.4955682, upper bound: 36.4957118
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 6, lower bound: -36.4958715, upper bound: 36.4959845
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 6, lower bound: -36.4970548, upper bound: 36.4973549
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 6, lower bound: -36.4975467, upper bound: 36.4977601
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 6, lower bound: -36.4961430, upper bound: 36.4962081
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.04
Output dim: 6, lower bound: -36.4965300, upper bound: 36.4965346

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -25.3202496, 18.2053394, -23.2091007, 16.6837540, -42.0039978, 41.4144402
1: -18.3395081, 18.2720432, -16.7842026, 16.7775993, -35.1171074, 35.0562439
2: -25.7739754, 17.5239296, -23.5941200, 16.0796623, -41.8536301, 41.1180496
3: -29.6329422, 14.6733255, -27.1541176, 13.4718266, -43.1047668, 41.8274422
4: -30.3738632, 17.5533257, -27.8705959, 16.0850220, -46.4588852, 45.4239197
5: -26.7107792, 15.6437016, -24.5137730, 14.3379574, -41.0487289, 40.1574707
6: -30.1578693, 14.9391203, -27.7162495, 13.6283426, -43.7862129, 42.6553612
7: -22.7493706, 22.2796650, -20.8393230, 20.4495163, -43.1988869, 43.1189804
8: -32.8561821, 16.1690884, -30.1353226, 14.8046417, -47.6608200, 46.3044052
9: -21.6963978, 21.9950066, -19.8906403, 20.1586037, -41.8550034, 41.8856468

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4971344, upper bound: 36.4968439
time: 7.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4971344, upper bound: 36.4968439
time: 10.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -25.3202496, 18.2053394, -23.7570133, 17.0723934, -42.3926392, 41.9623489
1: -18.3395081, 18.2720432, -17.1816635, 17.1663113, -35.5058212, 35.4537048
2: -25.7739754, 17.5239296, -24.1585579, 16.4491787, -42.2231522, 41.6824799
3: -29.6329422, 14.6733255, -27.7907715, 13.7832479, -43.4161911, 42.4640884
4: -30.3738632, 17.5533257, -28.5231457, 16.4542408, -46.8281021, 46.0764694
5: -26.7107792, 15.6437016, -25.0884323, 14.6764097, -41.3871880, 40.7321320
6: -30.1578693, 14.9391203, -28.3747330, 13.9489927, -44.1068611, 43.3138542
7: -22.7493706, 22.2796650, -21.3322258, 20.9263649, -43.6757355, 43.6118927
8: -32.8561821, 16.1690884, -30.8459702, 15.1462345, -48.0024185, 47.0150604
9: -21.6963978, 21.9950066, -20.3516388, 20.6292839, -42.3256836, 42.3466415

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4971345, upper bound: 36.4969146
time: 8.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4971345, upper bound: 36.4969147
time: 5.99 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -24.8111839, 17.8372879, -28.6291904, 20.5481834, -45.3593521, 46.4664764
1: -17.9620838, 17.9098415, -20.7558098, 20.5986404, -38.5607224, 38.6656494
2: -25.2448921, 17.1761055, -29.1427975, 19.7753525, -45.0202408, 46.3189011
3: -29.0296211, 14.3817596, -33.5036659, 16.5291519, -45.5587730, 47.8854256
4: -29.7682209, 17.1942215, -34.3550987, 19.8019409, -49.5701599, 51.5493202
5: -26.1785583, 15.3290939, -30.2259789, 17.6345253, -43.8130836, 45.5550728
6: -29.5686817, 14.6220589, -34.0285149, 16.8245296, -46.3932114, 48.6505699
7: -22.2865772, 21.8359604, -25.7015953, 25.1533852, -47.4399643, 47.5375519
8: -32.2019958, 15.8379002, -37.1919746, 18.2266521, -50.4286499, 53.0298767
9: -21.2563820, 21.5486012, -24.4976997, 24.8569794, -46.1133575, 46.0462914

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4889427, upper bound: 36.4890455
time: 4.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4884161, upper bound: 36.4884775
time: 6.47 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -24.8111839, 17.8372879, -29.1363792, 20.9108906, -45.7220573, 46.9736671
1: -17.9620838, 17.9098415, -21.1239662, 20.9575844, -38.9196701, 39.0338058
2: -25.2448921, 17.1761055, -29.6632805, 20.1197701, -45.3646584, 46.8393860
3: -29.0296211, 14.3817596, -34.0889015, 16.8171082, -45.8467293, 48.4706612
4: -29.7682209, 17.1942215, -34.9514618, 20.1458645, -49.9140854, 52.1456833
5: -26.1785583, 15.3290939, -30.7527027, 17.9518681, -44.1304245, 46.0817947
6: -29.5686817, 14.6220589, -34.6341133, 17.1315460, -46.7002258, 49.2561722
7: -22.2865772, 21.8359604, -26.1560345, 25.5930195, -47.8795967, 47.9919930
8: -32.2019958, 15.8379002, -37.8460045, 18.5496426, -50.7516365, 53.6839027
9: -21.2563820, 21.5486012, -24.9241676, 25.2911243, -46.5475082, 46.4727669

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4889428, upper bound: 36.4913728
time: 10.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4884161, upper bound: 36.4906445
time: 7.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -23.9371262, 17.1970825, -24.7387066, 17.7954140, -41.7325401, 41.9357910
1: -17.3124199, 17.2912521, -17.9190502, 17.8599777, -35.1723976, 35.2102966
2: -24.3421192, 16.5659676, -25.1771355, 17.1360683, -41.4781876, 41.7431030
3: -28.0022202, 13.8827324, -28.9569740, 14.3445644, -42.3467827, 42.8397064
4: -28.7457294, 16.5731564, -29.6746178, 17.1653023, -45.9110336, 46.2477722
5: -25.2835369, 14.7797403, -26.0969582, 15.2902641, -40.5737991, 40.8766937
6: -28.5909576, 14.0425768, -29.4518929, 14.6127911, -43.2037392, 43.4944687
7: -21.4927864, 21.0835342, -22.2283058, 21.7722740, -43.2650566, 43.3118401
8: -31.0806217, 15.2509584, -32.1001816, 15.8135033, -46.8941269, 47.3511391
9: -20.5031548, 20.7837429, -21.2091331, 21.4971275, -42.0002823, 41.9928741

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4792598, upper bound: 36.4798694
time: 10.07 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4778897, upper bound: 36.4783684
time: 57.95 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -24.8220692, 17.8380280, -25.3202496, 18.2053394, -43.0273933, 43.1582718
1: -17.9669228, 17.9202499, -18.3395081, 18.2720432, -36.2389679, 36.2597504
2: -25.2572556, 17.1706696, -25.7739754, 17.5239296, -42.7811813, 42.9446373
3: -29.0435638, 14.3876972, -29.6329422, 14.6733255, -43.7168884, 44.0206375
4: -29.7967110, 17.1909294, -30.3738632, 17.5533257, -47.3500366, 47.5647926
5: -26.2105141, 15.3283358, -26.7107792, 15.6437016, -41.8542175, 42.0391159
6: -29.6201782, 14.5928993, -30.1578693, 14.9391203, -44.5592995, 44.7507706
7: -22.2955475, 21.8540039, -22.7493706, 22.2796650, -44.5752029, 44.6033707
8: -32.2208405, 15.8253775, -32.8561821, 16.1690884, -48.3899307, 48.6815491
9: -21.2628174, 21.5562477, -21.6963978, 21.9950066, -43.2578201, 43.2526436

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4798620, upper bound: 36.4804338
time: 11.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4785028, upper bound: 36.4789563
time: 11.30 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -29.3937645, 21.0939808, -24.2528858, 17.4435883, -46.8373528, 45.3468590
1: -21.3041534, 21.1345825, -17.5585270, 17.5143013, -38.8184547, 38.6931038
2: -29.9331474, 20.2862988, -24.6718369, 16.8036156, -46.7367592, 44.9581375
3: -34.3793945, 16.9571953, -28.3814125, 14.0661440, -48.4455338, 45.3386078
4: -35.2552795, 20.3112850, -29.0973167, 16.8221760, -52.0774345, 49.4085999
5: -31.0142994, 18.1126976, -25.5894642, 14.9893951, -46.0036926, 43.7021599
6: -34.8995552, 17.2977943, -28.8907185, 14.3086224, -49.2081680, 46.1885147
7: -26.3947906, 25.8094864, -21.7863007, 21.3489895, -47.7437820, 47.5957870
8: -38.1685562, 18.7103233, -31.4761429, 15.4964418, -53.6649971, 50.1864662
9: -25.1480827, 25.5050240, -20.7890224, 21.0709343, -46.2190170, 46.2940445

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4888183, upper bound: 36.4890172
time: 26.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4882908, upper bound: 36.4884242
time: 10.46 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -30.2610035, 21.7219810, -24.8111839, 17.8372879, -48.0982895, 46.5331573
1: -21.9436512, 21.7475586, -17.9620838, 17.9098415, -39.8534927, 39.7096367
2: -30.8307724, 20.8763409, -25.2448921, 17.1761055, -48.0068779, 46.1212311
3: -35.3967896, 17.4510441, -29.0296211, 14.3817596, -49.7785416, 46.4806671
4: -36.2803345, 20.9143734, -29.7682209, 17.1942215, -53.4745522, 50.6825943
5: -31.9150505, 18.6543045, -26.1785583, 15.3290939, -47.2441444, 44.8328629
6: -35.8894081, 17.8466530, -29.5686817, 14.6220589, -50.5114632, 47.4153366
7: -27.1830673, 26.5601311, -22.2865772, 21.8359604, -49.0190277, 48.8467102
8: -39.2783508, 19.2779160, -32.2019958, 15.8379002, -55.1162491, 51.4799080
9: -25.8940830, 26.2599907, -21.2563820, 21.5486012, -47.4426842, 47.5163727

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4913509, upper bound: 36.4915246
time: 7.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4906446, upper bound: 36.4907113
time: 7.12 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -23.9371262, 17.1970825, -24.4690228, 17.5885658, -41.5256920, 41.6661072
1: -17.3124199, 17.2912521, -17.7143955, 17.6709976, -34.9834175, 35.0056419
2: -24.3421192, 16.5659676, -24.8970509, 16.9329281, -41.2750473, 41.4630203
3: -28.0022202, 13.8827324, -28.6422386, 14.1885567, -42.1907730, 42.5249710
4: -28.7457294, 16.5731564, -29.3781872, 16.9604244, -45.7061539, 45.9513435
5: -25.2835369, 14.7797403, -25.8397369, 15.1101332, -40.3936691, 40.6194763
6: -28.5909576, 14.0425768, -29.1857796, 14.3930960, -42.9840469, 43.2283554
7: -21.4927864, 21.0835342, -21.9804668, 21.5466728, -43.0394592, 43.0640030
8: -31.0806217, 15.2509584, -31.7605591, 15.6121273, -46.6927452, 47.0115166
9: -20.5031548, 20.7837429, -20.9723015, 21.2582264, -41.7613831, 41.7560425

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4811666, upper bound: 36.4817891
time: 8.77 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4797645, upper bound: 36.4802632
time: 9.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -24.8220692, 17.8380280, -24.9903774, 17.9583054, -42.7803650, 42.8283920
1: -17.9669228, 17.9202499, -18.0927162, 18.0417156, -36.0086365, 36.0129623
2: -25.2572556, 17.1706696, -25.4342766, 17.2852230, -42.5424805, 42.6049423
3: -29.0435638, 14.3876972, -29.2472229, 14.4849119, -43.5284729, 43.6349182
4: -29.7967110, 17.1909294, -29.9997368, 17.3122025, -47.1089134, 47.1906662
5: -26.2105141, 15.3283358, -26.3872337, 15.4319744, -41.6424828, 41.7155685
6: -29.6201782, 14.5928993, -29.8159714, 14.6964741, -44.3166504, 44.4088669
7: -22.2955475, 21.8540039, -22.4493523, 22.0019207, -44.2974625, 44.3033485
8: -32.2208405, 15.8253775, -32.4376831, 15.9359560, -48.1567955, 48.2630463
9: -21.2628174, 21.5562477, -21.4101429, 21.7058010, -42.9686165, 42.9663925

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4817239, upper bound: 36.4823315
time: 13.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4803646, upper bound: 36.4808332
time: 7.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -29.3937645, 21.0939808, -23.9729156, 17.2313976, -46.6251602, 45.0668945
1: -21.3041534, 21.1345825, -17.3477097, 17.3189430, -38.6230927, 38.4822845
2: -29.9331474, 20.2862988, -24.3819542, 16.5970612, -46.5301971, 44.6682434
3: -34.3793945, 16.9571953, -28.0551167, 13.9047861, -48.2841682, 45.0123062
4: -35.2552795, 20.3112850, -28.7879333, 16.6122818, -51.8675537, 49.0992203
5: -31.0142994, 18.1126976, -25.3217030, 14.8042974, -45.8185959, 43.4344025
6: -34.8995552, 17.2977943, -28.6136589, 14.0846081, -48.9841614, 45.9114532
7: -26.3947906, 25.8094864, -21.5296192, 21.1160355, -47.5108261, 47.3391037
8: -38.1685562, 18.7103233, -31.1243229, 15.2898674, -53.4584236, 49.8346405
9: -25.1480827, 25.5050240, -20.5442142, 20.8231716, -45.9712524, 46.0492401

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4893813, upper bound: 36.4895755
time: 17.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4888574, upper bound: 36.4889773
time: 7.78 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.38 seconds
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4971344, upper bound: 36.4968439
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4971344, upper bound: 36.4968439
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4971345, upper bound: 36.4969146
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4971345, upper bound: 36.4969147
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4889427, upper bound: 36.4890455
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4884161, upper bound: 36.4884775
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4889428, upper bound: 36.4913728
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4884161, upper bound: 36.4906445
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4792598, upper bound: 36.4798694
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4778897, upper bound: 36.4783684
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4798620, upper bound: 36.4804338
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4785028, upper bound: 36.4789563
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4888183, upper bound: 36.4890172
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4882908, upper bound: 36.4884242
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4913509, upper bound: 36.4915246
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4906446, upper bound: 36.4907113
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4811666, upper bound: 36.4817891
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4797645, upper bound: 36.4802632
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4817239, upper bound: 36.4823315
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4803646, upper bound: 36.4808332
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4893813, upper bound: 36.4895755
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 26.38
Output dim: 6, lower bound: -36.4888574, upper bound: 36.4889773
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.38
Output dim: 6, lower bound: -36.4965300, upper bound: 36.4965346
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=47.04602813720703
rel_dist={6: [-36.50308451104452, 36.50308451104452]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5027146, upper bound: 36.5026891
time: 8.42 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5029736, upper bound: 36.5029736
time: 33.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 42.11 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 42.11
Output dim: 6, lower bound: -36.5027146, upper bound: 36.5026891
IS_A2, status: Status.UNKNOWN, split count: 1, time: 42.11
Output dim: 6, lower bound: -36.5029736, upper bound: 36.5029736

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -26.1358681, 18.7964439, -24.4370899, 17.5688992, -43.7047653, 43.2335320
1: -18.9409447, 18.8499107, -17.6879787, 17.6530685, -36.5940132, 36.5378838
2: -26.6191959, 18.0804043, -24.8667488, 16.9156952, -43.5348892, 42.9471474
3: -30.5916786, 15.1372766, -28.5993252, 14.1746349, -44.7663116, 43.7366028
4: -31.3408966, 18.1206932, -29.3267765, 16.9394283, -48.2803192, 47.4474716
5: -27.5593796, 16.1499977, -25.7946320, 15.1023073, -42.6616859, 41.9446297
6: -31.0921783, 15.4537849, -29.1555595, 14.3904133, -45.4825897, 44.6093445
7: -23.4900723, 22.9867249, -21.9539165, 21.5185585, -45.0086250, 44.9406357
8: -33.9014206, 16.6999149, -31.7155628, 15.5994158, -49.5008278, 48.4154663
9: -22.3984489, 22.7051811, -20.9425259, 21.2289124, -43.6273613, 43.6477013

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4962187, upper bound: 36.4960776
time: 8.58 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4957418, upper bound: 36.4956964
time: 13.17 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -25.8575134, 18.5841770, -26.0606098, 18.7303963, -44.5879097, 44.6447868
1: -18.7319984, 18.6548462, -18.8814392, 18.7984390, -37.5304375, 37.5362854
2: -26.3309937, 17.8734856, -26.5410500, 18.0111389, -44.3421326, 44.4145355
3: -30.2666035, 14.9785585, -30.5049171, 15.0938816, -45.3604851, 45.4834747
4: -31.0305214, 17.9139519, -31.2722855, 18.0541286, -49.0846481, 49.1862335
5: -27.2929516, 15.9670887, -27.5056438, 16.0924282, -43.3853760, 43.4727325
6: -30.8138809, 15.2353678, -31.0472965, 15.3605757, -46.1744576, 46.2826614
7: -23.2353802, 22.7546539, -23.4190865, 22.9307919, -46.1661720, 46.1737366
8: -33.5508270, 16.4978981, -33.8127441, 16.6289234, -50.1797485, 50.3106308
9: -22.1546326, 22.4615669, -22.3287086, 22.6381245, -44.7927551, 44.7902756

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4968275, upper bound: 36.4967054
time: 10.27 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4963376, upper bound: 36.4963376
time: 8.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 20.09 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 20.09
Output dim: 6, lower bound: -36.4962187, upper bound: 36.4960776
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 20.09
Output dim: 6, lower bound: -36.4957418, upper bound: 36.4956964
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.09
Output dim: 6, lower bound: -36.4968275, upper bound: 36.4967054
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.09
Output dim: 6, lower bound: -36.4963376, upper bound: 36.4963376

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -25.5087280, 18.3436127, -23.5349159, 16.9177170, -42.4264450, 41.8785286
1: -18.4778481, 18.4044895, -17.0190887, 17.0098858, -35.4877281, 35.4235764
2: -25.9685535, 17.6526508, -23.9305439, 16.3009529, -42.2695084, 41.5831871
3: -29.8519650, 14.7800922, -27.5300694, 13.6588717, -43.5108376, 42.3101578
4: -30.5961246, 17.6821747, -28.2500725, 16.3055344, -46.9016571, 45.9322472
5: -26.9069042, 15.7614250, -24.8476562, 14.5455017, -41.4524002, 40.6090813
6: -30.3729858, 15.0601597, -28.1056480, 13.8330212, -44.2059975, 43.1658096
7: -22.9203796, 22.4426689, -21.1337738, 20.7314987, -43.6518784, 43.5764427
8: -33.0987320, 16.2915611, -30.5527306, 15.0144892, -48.1132202, 46.8442917
9: -21.8576431, 22.1578045, -20.1644096, 20.4378586, -42.2954979, 42.3222122

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4958521, upper bound: 36.4957078
time: 16.63 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4962191, upper bound: 36.4960776
time: 10.62 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -24.9455070, 17.9369202, -28.8416481, 20.7046738, -45.6501808, 46.7785606
1: -18.0604534, 18.0044479, -20.9100914, 20.7528343, -38.8132858, 38.9145393
2: -25.3828163, 17.2697334, -29.3599243, 19.9255447, -45.3083611, 46.6296577
3: -29.1846771, 14.4573383, -33.7438736, 16.6511993, -45.8358765, 48.2012100
4: -29.9276047, 17.2841301, -34.5936775, 19.9503975, -49.8779984, 51.8778076
5: -26.3191338, 15.4135456, -30.4414635, 17.7751846, -44.0943031, 45.8550110
6: -29.7234039, 14.7093983, -34.2997055, 16.9631157, -46.6865158, 49.0091019
7: -22.4080391, 21.9527893, -25.8917866, 25.3393478, -47.7473831, 47.8445740
8: -32.3763237, 15.9254971, -37.4657860, 18.3662128, -50.7425156, 53.3912811
9: -21.3704300, 21.6637630, -24.6729355, 25.0383053, -46.4087372, 46.3367004

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4954250, upper bound: 36.4953692
time: 10.58 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4957418, upper bound: 36.4956964
time: 8.45 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -25.2274895, 18.1303024, -25.1237793, 18.0552826, -43.2827721, 43.2540817
1: -18.2663784, 18.2077217, -18.1891193, 18.1333218, -36.3997002, 36.3968430
2: -25.6773186, 17.4458847, -25.5689774, 17.3753376, -43.0526543, 43.0148544
3: -29.5222397, 14.6189737, -29.3980103, 14.5593090, -44.0815506, 44.0169830
4: -30.2798481, 17.4736729, -30.1558952, 17.3995724, -47.6794167, 47.6295624
5: -26.6345787, 15.5782452, -26.5266228, 15.5141602, -42.1487350, 42.1048584
6: -30.0880280, 14.8440495, -29.9683247, 14.7785473, -44.8665771, 44.8123703
7: -22.6632957, 22.2066612, -22.5682793, 22.1159515, -44.7792435, 44.7749329
8: -32.7418137, 16.0884800, -32.6098099, 16.0201225, -48.7619362, 48.6982880
9: -21.6116714, 21.9106293, -21.5212231, 21.8189030, -43.4305725, 43.4318504

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4964328, upper bound: 36.4962999
time: 11.08 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4968275, upper bound: 36.4967054
time: 11.07 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -24.6449223, 17.7113171, -30.5616245, 21.9391823, -46.5840988, 48.2729416
1: -17.8358841, 17.7946739, -22.1653042, 21.9605179, -39.7964020, 39.9599762
2: -25.0723114, 17.0522270, -31.1415405, 21.0794888, -46.1518021, 48.1937675
3: -28.8326263, 14.2855053, -35.7500458, 17.6215324, -46.4541512, 50.0355453
4: -29.5866508, 17.0642433, -36.6374207, 21.1226749, -50.7093277, 53.7016640
5: -26.0264435, 15.2197952, -32.2285652, 18.8409557, -44.8674011, 47.4483566
6: -29.4165173, 14.4837723, -36.2324486, 18.0342388, -47.4507523, 50.7162209
7: -22.1341286, 21.7006836, -27.4554749, 26.8201294, -48.9542580, 49.1561546
8: -31.9949608, 15.7110882, -39.6640968, 19.4727097, -51.4676704, 55.3751831
9: -21.1086998, 21.3996868, -26.1526222, 26.5213737, -47.6300697, 47.5523071

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4959952, upper bound: 36.4959823
time: 8.10 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4957418, upper bound: 36.4963376
time: 9.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.79 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.79
Output dim: 6, lower bound: -36.4958521, upper bound: 36.4957078
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.79
Output dim: 6, lower bound: -36.4962191, upper bound: 36.4960776
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.79
Output dim: 6, lower bound: -36.4954250, upper bound: 36.4953692
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.79
Output dim: 6, lower bound: -36.4957418, upper bound: 36.4956964
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.79
Output dim: 6, lower bound: -36.4964328, upper bound: 36.4962999
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.79
Output dim: 6, lower bound: -36.4968275, upper bound: 36.4967054
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.79
Output dim: 6, lower bound: -36.4959952, upper bound: 36.4959823
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.79
Output dim: 6, lower bound: -36.4957418, upper bound: 36.4963376

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -24.2854195, 17.4666824, -21.8366127, 15.6841125, -39.9695320, 39.3032951
1: -17.5825348, 17.5368690, -15.7620392, 15.8003826, -33.3829117, 33.2989044
2: -24.7060776, 16.8247375, -22.1720810, 15.1385746, -39.8446465, 38.9968109
3: -28.4200935, 14.0850830, -25.5308208, 12.6881847, -41.1082764, 39.6159058
4: -29.1349907, 16.8460197, -26.2332516, 15.1202803, -44.2552719, 43.0792656
5: -25.6223907, 15.0092764, -23.0662117, 13.4902039, -39.1125946, 38.0754890
6: -28.9263020, 14.3292513, -26.1268005, 12.7753563, -41.7016602, 40.4560509
7: -21.8161144, 21.3764725, -19.5896835, 19.2525921, -41.0687027, 40.9661560
8: -31.5164490, 15.5172567, -28.3610401, 13.9114552, -45.4279022, 43.8782959
9: -20.8176403, 21.0999641, -18.7039452, 18.9536877, -39.7713165, 39.8039093

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4771829, upper bound: 36.4769773
time: 11.47 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4766535, upper bound: 36.4764916
time: 30.42 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -24.8642368, 17.8749733, -23.0879135, 16.5942268, -41.4584656, 40.9628868
1: -18.0013523, 17.9470634, -16.6887474, 16.6928577, -34.6942101, 34.6358070
2: -25.3004799, 17.2110176, -23.4686279, 15.9964046, -41.2968826, 40.6796455
3: -29.0926971, 14.4125118, -27.0041809, 13.4041691, -42.4968643, 41.4166946
4: -29.8305779, 17.2325478, -27.7187481, 15.9941950, -45.8247643, 44.9512939
5: -26.2331543, 15.3614855, -24.3790283, 14.2695961, -40.5027390, 39.7405128
6: -29.6284332, 14.6547937, -27.5863686, 13.5558748, -43.1843033, 42.2411613
7: -22.3350143, 21.8813972, -20.7283802, 20.3426666, -42.6776810, 42.6097755
8: -32.2690048, 15.8719082, -29.9770279, 14.7254372, -46.9944305, 45.8489380
9: -21.3026638, 21.5957031, -19.7809029, 20.0476379, -41.3502960, 41.3766022

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4776252, upper bound: 36.4773872
time: 11.22 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4770997, upper bound: 36.4769166
time: 12.55 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -23.7589493, 17.0853939, -27.1869335, 19.5033073, -43.2622566, 44.2723274
1: -17.1916142, 17.1627235, -19.6837559, 19.5763741, -36.7679863, 36.8464813
2: -24.1578369, 16.4657478, -27.6493530, 18.7915001, -42.9493370, 44.1151009
3: -27.7959366, 13.7829609, -31.7961273, 15.7060461, -43.5019836, 45.5790863
4: -28.5102692, 16.4735069, -32.6326981, 18.7918854, -47.3021545, 49.1061935
5: -25.0726624, 14.6833124, -28.7120667, 16.7445126, -41.8171768, 43.3953743
6: -28.3190079, 13.9997635, -32.3811569, 15.9284983, -44.2475052, 46.3809204
7: -21.3363991, 20.9182587, -24.3896980, 23.8985443, -45.2349358, 45.3079491
8: -30.8405476, 15.1739864, -35.3356781, 17.2870560, -48.1276016, 50.5096588
9: -20.3617573, 20.6375313, -23.2510834, 23.5928288, -43.9545860, 43.8885994

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4880767, upper bound: 36.4880034
time: 12.57 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4878634, upper bound: 36.4878139
time: 6.44 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -24.2969856, 17.4651432, -28.4264088, 20.4034500, -44.7004356, 45.8915482
1: -17.5808506, 17.5439701, -20.6024246, 20.4581699, -38.0390167, 38.1463890
2: -24.7105446, 16.8250637, -28.9308853, 19.6418667, -44.3524094, 45.7559509
3: -28.4204865, 14.0873566, -33.2549896, 16.4143562, -44.8348427, 47.3423462
4: -29.1565857, 16.8316879, -34.1015549, 19.6597786, -48.8163643, 50.9332428
5: -25.6406021, 15.0115080, -30.0076122, 17.5173092, -43.1579132, 45.0191193
6: -28.9730797, 14.3020315, -33.8197632, 16.7032547, -45.6763344, 48.1217957
7: -21.8190155, 21.3876591, -25.5148430, 24.9780731, -46.7970886, 46.9025002
8: -31.5408382, 15.5032387, -36.9319839, 18.0956802, -49.6365089, 52.4352188
9: -20.8119965, 21.0978756, -24.3161201, 24.6754456, -45.4874420, 45.4139938

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4908030, upper bound: 36.4907319
time: 11.52 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4904975, upper bound: 36.4904708
time: 11.43 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -24.0093632, 17.2573605, -23.3437691, 16.7663326, -40.7756958, 40.6011276
1: -17.3745155, 17.3444881, -16.8733406, 16.8694763, -34.2439804, 34.2178268
2: -24.4204216, 16.6207352, -23.7286301, 16.1599274, -40.5803490, 40.3493652
3: -28.0983887, 13.9261465, -27.3039703, 13.5439224, -41.6423111, 41.2301178
4: -28.8303261, 16.6388817, -28.0425301, 16.1582394, -44.9885635, 44.6814117
5: -25.3588791, 14.8266735, -24.6634369, 14.4109383, -39.7698174, 39.4901047
6: -28.6546097, 14.1076412, -27.9023685, 13.6699591, -42.3245697, 42.0100098
7: -21.5631618, 21.1469612, -20.9536743, 20.5677109, -42.1308746, 42.1006355
8: -31.1698914, 15.3133001, -30.3181286, 14.8642807, -46.0341721, 45.6314278
9: -20.5762939, 20.8556900, -19.9934196, 20.2653732, -40.8416672, 40.8491096

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4793655, upper bound: 36.4790668
time: 8.35 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4787546, upper bound: 36.4785553
time: 9.42 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -24.5388031, 17.6322479, -24.6579971, 17.7185574, -42.2573547, 42.2902451
1: -17.7580490, 17.7204094, -17.8454552, 17.8039951, -35.5620422, 35.5658646
2: -24.9656067, 16.9776611, -25.0876064, 17.0589581, -42.0245667, 42.0652618
3: -28.7126312, 14.2267780, -28.8505478, 14.2940588, -43.0066872, 43.0773239
4: -29.4614449, 16.9951401, -29.6023254, 17.0762978, -46.5377426, 46.5974655
5: -25.9141846, 15.1530781, -26.0396919, 15.2266922, -41.1408691, 41.1927719
6: -29.2924881, 14.4161930, -29.4312096, 14.4889975, -43.7814827, 43.8473930
7: -22.0391350, 21.6080704, -22.1461678, 21.7113972, -43.7505188, 43.7542381
8: -31.8564835, 15.6422052, -32.0113945, 15.7181835, -47.5746651, 47.6535988
9: -21.0207615, 21.3100471, -21.1215687, 21.4128075, -42.4335670, 42.4316177

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4798401, upper bound: 36.4795404
time: 24.67 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4792312, upper bound: 36.4790225
time: 8.19 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -23.4747105, 16.8718815, -28.7995625, 20.6630650, -44.1377754, 45.6714401
1: -16.9787025, 16.9639244, -20.8661594, 20.7141361, -37.6928406, 37.8300858
2: -23.8648415, 16.2585373, -29.3180618, 19.8814049, -43.7462349, 45.5765991
3: -27.4634972, 13.6195650, -33.6822968, 16.6185417, -44.0820389, 47.3018570
4: -28.1934471, 16.2616673, -34.5538483, 19.8975620, -48.0910034, 50.8155060
5: -24.7990303, 14.4968739, -30.3986626, 17.7409248, -42.5399551, 44.8955383
6: -28.0339642, 13.7763901, -34.2229042, 16.9189835, -44.9529495, 47.9992943
7: -21.0764656, 20.6821404, -25.8541317, 25.2958660, -46.3723297, 46.5362663
8: -30.4823685, 14.9671106, -37.4096413, 18.3204136, -48.8027763, 52.3767509
9: -20.1135101, 20.3859024, -24.6367264, 24.9877148, -45.1012230, 45.0226288

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4886838, upper bound: 36.4886145
time: 14.33 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4884830, upper bound: 36.4884432
time: 14.15 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -23.9874897, 17.2350674, -30.1193333, 21.6188698, -45.6063576, 47.3543930
1: -17.3496513, 17.3283463, -21.8393288, 21.6475849, -38.9972382, 39.1676674
2: -24.3927078, 16.6041069, -30.6841736, 20.7795010, -45.1722031, 47.2882805
3: -28.0587616, 13.9108019, -35.2303886, 17.3702698, -45.4290314, 49.1411896
4: -28.8055420, 16.6055470, -36.1138916, 20.8153152, -49.6208572, 52.7194366
5: -25.3374729, 14.8134098, -31.7693253, 18.5654488, -43.9029236, 46.5827332
6: -28.6534824, 14.0746708, -35.7295074, 17.7538853, -46.4073639, 49.8041687
7: -21.5375996, 21.1286736, -27.0536652, 26.4378166, -47.9754181, 48.1823387
8: -31.1483784, 15.2847824, -39.0988617, 19.1840057, -50.3323669, 54.3836288
9: -20.5442123, 20.8254719, -25.7718563, 26.1363277, -46.6805420, 46.5973206

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4913575, upper bound: 36.4912974
time: 9.33 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4910350, upper bound: 36.4910350
time: 17.09 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 27.53 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 27.53
Output dim: 6, lower bound: -36.4771829, upper bound: 36.4769773
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 27.53
Output dim: 6, lower bound: -36.4766535, upper bound: 36.4764916
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 27.53
Output dim: 6, lower bound: -36.4776252, upper bound: 36.4773872
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 27.53
Output dim: 6, lower bound: -36.4770997, upper bound: 36.4769166
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 27.53
Output dim: 6, lower bound: -36.4880767, upper bound: 36.4880034
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 27.53
Output dim: 6, lower bound: -36.4878634, upper bound: 36.4878139
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 6, lower bound: -36.4908030, upper bound: 36.4907319
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 6, lower bound: -36.4904975, upper bound: 36.4904708
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 27.53
Output dim: 6, lower bound: -36.4793655, upper bound: 36.4790668
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 27.53
Output dim: 6, lower bound: -36.4787546, upper bound: 36.4785553
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 27.53
Output dim: 6, lower bound: -36.4798401, upper bound: 36.4795404
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 27.53
Output dim: 6, lower bound: -36.4792312, upper bound: 36.4790225
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 27.53
Output dim: 6, lower bound: -36.4886838, upper bound: 36.4886145
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 27.53
Output dim: 6, lower bound: -36.4884830, upper bound: 36.4884432
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 6, lower bound: -36.4913575, upper bound: 36.4912974
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.53
Output dim: 6, lower bound: -36.4910350, upper bound: 36.4910350

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -23.4477997, 16.8413315, -27.2813892, 19.5640182, -43.0118179, 44.1227188
1: -16.9432297, 16.9336967, -19.7409401, 19.6352749, -36.5785065, 36.6746368
2: -23.8227997, 16.2358208, -27.7357368, 18.8505936, -42.6733894, 43.9715538
3: -27.4118729, 13.5965786, -31.8944893, 15.7520561, -43.1639290, 45.4910660
4: -28.1530132, 16.2195511, -32.7513809, 18.8312454, -46.9842606, 48.9709320
5: -24.7566013, 14.4753447, -28.8179893, 16.7916107, -41.5482101, 43.2933350
6: -28.0029850, 13.7501793, -32.5179062, 15.9577227, -43.9607010, 46.2680855
7: -21.0398312, 20.6464081, -24.4649563, 23.9794998, -45.0193329, 45.1113548
8: -30.4487534, 14.9352732, -35.4619751, 17.3280830, -47.7768326, 50.3972473
9: -20.0720100, 20.3480072, -23.3184967, 23.6624451, -43.7344551, 43.6664886

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4712945, upper bound: 36.4712998
time: 54.19 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4701246, upper bound: 36.4700566
time: 15.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.5646076, 16.1841240, -32.5815506, 23.3254662, -45.8900757, 48.7656746
1: -16.2736588, 16.2903576, -23.6210709, 23.3904800, -39.6641312, 39.9114304
2: -22.8965111, 15.6161308, -33.2023773, 22.4070568, -45.3035622, 48.8185043
3: -26.3646660, 13.0783939, -38.1631966, 18.7391396, -45.1038055, 51.2415924
4: -27.1223049, 15.5680447, -39.2386742, 22.4054604, -49.5277634, 54.8067169
5: -23.8474293, 13.8994865, -34.5294952, 19.9340000, -43.7814293, 48.4289780
6: -27.0063457, 13.1481352, -38.8761559, 18.9096870, -45.9160309, 52.0242920
7: -20.2217693, 19.8812790, -29.2520046, 28.6258411, -48.8476105, 49.1332855
8: -29.3136063, 14.3280582, -42.4361992, 20.5559082, -49.8695068, 56.7642555
9: -19.2991886, 19.5599766, -27.8375683, 28.2775078, -47.5766945, 47.3975449

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4709048, upper bound: 36.4709556
time: 18.37 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4698182, upper bound: 36.4697796
time: 6.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -23.1756973, 16.6401882, -28.9044838, 20.7337074, -43.9094048, 45.5446701
1: -16.7406197, 16.7431316, -20.9333267, 20.7817535, -37.5223732, 37.6764526
2: -23.5447979, 16.0429211, -29.4164505, 19.9484749, -43.4932632, 45.4593697
3: -27.0940838, 13.4410095, -33.7971497, 16.6712017, -43.7652855, 47.2381592
4: -27.8458710, 16.0204678, -34.6879768, 19.9463692, -47.7922401, 50.7084198
5: -24.4917831, 14.2999926, -30.5194111, 17.7951450, -42.2869263, 44.8194046
6: -27.7238197, 13.5489893, -34.3783951, 16.9535942, -44.6774139, 47.9273834
7: -20.7913055, 20.4215775, -25.9394341, 25.3888817, -46.1801872, 46.3610115
8: -30.1040115, 14.7431326, -37.5533295, 18.3688297, -48.4728317, 52.2964516
9: -19.8365784, 20.1069393, -24.7149258, 25.0680962, -44.9046745, 44.8218536

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4734583, upper bound: 36.4735100
time: 9.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4726472, upper bound: 36.4726047
time: 11.55 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.3104973, 15.9978809, -34.2591171, 24.5411053, -46.8516006, 50.2569885
1: -16.0865250, 16.1114197, -24.8523731, 24.5796967, -40.6662216, 40.9637909
2: -22.6384830, 15.4369707, -34.9431992, 23.5468388, -46.1853142, 50.3801689
3: -26.0698338, 12.9336338, -40.1340218, 19.6895542, -45.7593880, 53.0676575
4: -26.8317680, 15.3865566, -41.2408371, 23.5573101, -50.3890762, 56.6273956
5: -23.5994568, 13.7375393, -36.2844810, 20.9791985, -44.5786552, 50.0220184
6: -26.7435265, 12.9635563, -40.7892685, 19.9536552, -46.6971817, 53.7528191
7: -19.9894714, 19.6728134, -30.7783604, 30.0861111, -50.0755768, 50.4511719
8: -28.9910221, 14.1521282, -44.6019287, 21.6395969, -50.6306152, 58.7540474
9: -19.0791492, 19.3354073, -29.2844982, 29.7301064, -48.8092575, 48.6199036

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4731206, upper bound: 36.4732325
time: 12.34 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4723123, upper bound: 36.4723123
time: 25.51 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 38.96 seconds
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 38.96
Output dim: 6, lower bound: -36.4712945, upper bound: 36.4712998
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 38.96
Output dim: 6, lower bound: -36.4701246, upper bound: 36.4700566
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 38.96
Output dim: 6, lower bound: -36.4709048, upper bound: 36.4709556
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 38.96
Output dim: 6, lower bound: -36.4698182, upper bound: 36.4697796
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 38.96
Output dim: 6, lower bound: -36.4734583, upper bound: 36.4735100
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 38.96
Output dim: 6, lower bound: -36.4726472, upper bound: 36.4726047
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 38.96
Output dim: 6, lower bound: -36.4731206, upper bound: 36.4732325
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 38.96
Output dim: 6, lower bound: -36.4723123, upper bound: 36.4723123
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=47.04602813720703
rel_dist={6: [-36.50297363759564, 36.50297363759566]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5027913, upper bound: 36.5027495
time: 12.42 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5030335, upper bound: 36.5030335
time: 8.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 20.84 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 20.84
Output dim: 6, lower bound: -36.5027913, upper bound: 36.5027495
IS_A2, status: Status.UNKNOWN, split count: 1, time: 20.84
Output dim: 6, lower bound: -36.5030335, upper bound: 36.5030335

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -26.1358681, 18.7964439, -24.9690704, 17.9505310, -44.0863876, 43.7655144
1: -18.9409447, 18.8499107, -18.0802631, 18.0291138, -36.9700546, 36.9301720
2: -26.6191959, 18.0804043, -25.4159069, 17.2755489, -43.8947449, 43.4963074
3: -30.5916786, 15.1372766, -29.2248077, 14.4764729, -45.0681496, 44.3620758
4: -31.3408966, 18.1206932, -29.9647942, 17.3062630, -48.6471558, 48.0854874
5: -27.5593796, 16.1499977, -26.3568649, 15.4271374, -42.9865112, 42.5068626
6: -31.0921783, 15.4537849, -29.7788143, 14.7075138, -45.7996902, 45.2325974
7: -23.4900723, 22.9867249, -22.4344864, 21.9826641, -45.4727287, 45.4212074
8: -33.9014206, 16.6999149, -32.4047279, 15.9368944, -49.8383141, 49.1046410
9: -22.3984489, 22.7051811, -21.3972893, 21.6916771, -44.0901260, 44.1024704

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4967233, upper bound: 36.4965081
time: 9.66 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4958755, upper bound: 36.4957907
time: 8.70 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -25.8575134, 18.5841770, -26.2341728, 18.8556137, -44.7131271, 44.8183517
1: -18.7319984, 18.6548462, -19.0091763, 18.9212265, -37.6532249, 37.6640244
2: -26.3309937, 17.8734856, -26.7207031, 18.1291504, -44.4601364, 44.5941887
3: -30.2666035, 14.9785585, -30.7083549, 15.1922588, -45.4588623, 45.6869125
4: -31.0305214, 17.9139519, -31.4785690, 18.1739960, -49.2045174, 49.3925209
5: -27.2929516, 15.9670887, -27.6869698, 16.1996384, -43.4925919, 43.6540604
6: -30.8138809, 15.2353678, -31.2463608, 15.4682808, -46.2821579, 46.4817276
7: -23.2353802, 22.7546539, -23.5760422, 23.0810623, -46.3164444, 46.3306885
8: -33.5508270, 16.4978981, -34.0361862, 16.7410812, -50.2919044, 50.5340805
9: -22.1546326, 22.4615669, -22.4774780, 22.7888145, -44.9434471, 44.9390450

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 153

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4973404, upper bound: 36.4971636
time: 10.62 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4964495, upper bound: 36.4964495
time: 22.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 34.28 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 34.28
Output dim: 6, lower bound: -36.4967233, upper bound: 36.4965081
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 34.28
Output dim: 6, lower bound: -36.4958755, upper bound: 36.4957907
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 34.28
Output dim: 6, lower bound: -36.4973404, upper bound: 36.4971636
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 34.28
Output dim: 6, lower bound: -36.4964495, upper bound: 36.4964495

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -25.7610016, 18.5257874, -24.0595169, 17.2937660, -43.0547676, 42.5853043
1: -18.6642933, 18.5836868, -17.4059143, 17.3817062, -36.0459976, 35.9896011
2: -26.2305069, 17.8248138, -24.4718246, 16.6568813, -42.8873825, 42.2966385
3: -30.1498642, 14.9240150, -28.1470871, 13.9565849, -44.1064491, 43.0711021
4: -30.8959808, 17.8588486, -28.8791008, 16.6676712, -47.5636520, 46.7379494
5: -27.1700344, 15.9176102, -25.4022274, 14.8658047, -42.0358391, 41.3198357
6: -30.6634426, 15.2179432, -28.7214928, 14.1454716, -44.8089104, 43.9394302
7: -23.1497383, 22.6620464, -21.6080322, 21.1887798, -44.3385162, 44.2700768
8: -33.4225044, 16.4558239, -31.2326298, 15.3467541, -48.7692566, 47.6884460
9: -22.0752048, 22.3781738, -20.6126213, 20.8945084, -42.9697037, 42.9907913

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4963788, upper bound: 36.4961105
time: 13.70 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4967233, upper bound: 36.4965081
time: 9.87 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -25.2417259, 18.1513119, -29.4016857, 21.1050701, -46.3467941, 47.5529976
1: -18.2801075, 18.2151985, -21.3201447, 21.1458187, -39.4259224, 39.5353394
2: -25.6907063, 17.4719238, -29.9377880, 20.3014030, -45.9921036, 47.4097137
3: -29.5356560, 14.6269855, -34.4016991, 16.9690399, -46.5046959, 49.0286865
4: -30.2798805, 17.4930458, -35.2634277, 20.3327980, -50.6126785, 52.7564735
5: -26.6290054, 15.5966835, -31.0265446, 18.1188145, -44.7478180, 46.6232300
6: -30.0664120, 14.8938122, -34.9360504, 17.3052979, -47.3717079, 49.8298569
7: -22.6773968, 22.2110424, -26.3980846, 25.8227425, -48.5001373, 48.6091270
8: -32.7572250, 16.1183281, -38.1831360, 18.7260780, -51.4832954, 54.3014565
9: -21.6264515, 21.9233932, -25.1530361, 25.5236397, -47.1500778, 47.0764313

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4955801, upper bound: 36.4954752
time: 11.37 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4958755, upper bound: 36.4957907
time: 23.84 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -25.4825478, 18.3140793, -25.2931137, 18.1771030, -43.6596451, 43.6071930
1: -18.4550076, 18.3887787, -18.3139172, 18.2529373, -36.7079391, 36.7026978
2: -25.9418831, 17.6190872, -25.7440071, 17.4902115, -43.4320946, 43.3630943
3: -29.8238010, 14.7646036, -29.5968895, 14.6556263, -44.4794273, 44.3614845
4: -30.5839558, 17.6519699, -30.3575993, 17.5166092, -48.1005592, 48.0095673
5: -26.9013405, 15.7355251, -26.7043304, 15.6184044, -42.5197449, 42.4398575
6: -30.3824997, 15.0020924, -30.1639309, 14.8823156, -45.2648163, 45.1660233
7: -22.8947468, 22.4287109, -22.7212219, 22.2630882, -45.1578293, 45.1499290
8: -33.0696106, 16.2541752, -32.8285904, 16.1292686, -49.1988640, 49.0827599
9: -21.8314095, 22.1337910, -21.6661682, 21.9662781, -43.7976761, 43.7999573

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4969102, upper bound: 36.4967016
time: 15.04 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4973404, upper bound: 36.4971636
time: 9.14 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -24.9502773, 17.9313622, -30.7678070, 22.0939617, -47.0442352, 48.6991692
1: -18.0618725, 18.0117588, -22.3170185, 22.1016350, -40.1635056, 40.3287773
2: -25.3892784, 17.2593365, -31.3688984, 21.2411346, -46.6304131, 48.6282272
3: -29.1946030, 14.4601622, -35.9780655, 17.7334423, -46.9280472, 50.4382095
4: -29.9509220, 17.2788849, -36.8684959, 21.2641258, -51.2150497, 54.1473770
5: -26.3465290, 15.4077988, -32.4304848, 18.9666176, -45.3131409, 47.8382835
6: -29.7705116, 14.6721821, -36.4620934, 18.1821156, -47.9526253, 51.1342697
7: -22.4114952, 21.9667015, -27.6415749, 26.9933548, -49.4048500, 49.6082764
8: -32.3878365, 15.9088659, -39.9255600, 19.6169510, -52.0047874, 55.8344116
9: -21.3722935, 21.6673965, -26.3233147, 26.6942196, -48.0665131, 47.9907074

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4961144, upper bound: 36.4960707
time: 7.03 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4964495, upper bound: 36.4964495
time: 11.90 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.04 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.04
Output dim: 6, lower bound: -36.4963788, upper bound: 36.4961105
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.04
Output dim: 6, lower bound: -36.4967233, upper bound: 36.4965081
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.04
Output dim: 6, lower bound: -36.4955801, upper bound: 36.4954752
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.04
Output dim: 6, lower bound: -36.4958755, upper bound: 36.4957907
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.04
Output dim: 6, lower bound: -36.4969102, upper bound: 36.4967016
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.04
Output dim: 6, lower bound: -36.4973404, upper bound: 36.4971636
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.04
Output dim: 6, lower bound: -36.4961144, upper bound: 36.4960707
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.04
Output dim: 6, lower bound: -36.4964495, upper bound: 36.4964495

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -24.5300064, 17.6440792, -22.7809830, 16.3668060, -40.8968124, 40.4250641
1: -17.7641335, 17.7112694, -16.4600983, 16.4726353, -34.2367706, 34.1713676
2: -24.9602413, 16.9927101, -23.1494675, 15.7827988, -40.7430420, 40.1421776
3: -28.7098579, 14.2251148, -26.6427250, 13.2266808, -41.9365387, 40.8678398
4: -29.4261990, 17.0183353, -27.3611736, 15.7752600, -45.2014542, 44.3795090
5: -25.8785210, 15.1609125, -24.0626984, 14.0726233, -39.9511375, 39.2236099
6: -29.2100525, 14.4821835, -27.2341900, 13.3492937, -42.5593376, 41.7163658
7: -22.0385475, 21.5900688, -20.4472141, 20.0763626, -42.1149101, 42.0372849
8: -31.8315258, 15.6770840, -29.5852184, 14.5167446, -46.3482704, 45.2622986
9: -21.0289154, 21.3142891, -19.5146885, 19.7779045, -40.8068199, 40.8289795

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4785921, upper bound: 36.4781531
time: 18.37 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4775561, upper bound: 36.4772231
time: 10.41 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -25.1105251, 18.0534248, -23.7928772, 17.1010380, -42.2115631, 41.8463020
1: -18.1839237, 18.1225662, -17.2088852, 17.1925468, -35.3764725, 35.3314514
2: -25.5561771, 17.3799973, -24.1962280, 16.4752998, -42.0314789, 41.5762253
3: -29.3844795, 14.5533714, -27.8331699, 13.8047237, -43.1891975, 42.3865356
4: -30.1239967, 17.4057808, -28.5616531, 16.4821434, -46.6061363, 45.9674339
5: -26.4910984, 15.5138960, -25.1225185, 14.7014103, -41.1925087, 40.6364136
6: -29.9143963, 14.8083153, -28.4115448, 13.9808826, -43.8952789, 43.2198563
7: -22.5587959, 22.0964966, -21.3661995, 20.9567719, -43.5155678, 43.4626961
8: -32.5860939, 16.0324135, -30.8889751, 15.1747169, -47.7608109, 46.9213829
9: -21.5153141, 21.8113422, -20.3838596, 20.6617527, -42.1770668, 42.1952019

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4791285, upper bound: 36.4786558
time: 7.68 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4780822, upper bound: 36.4777395
time: 9.95 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -24.0446167, 17.2926483, -28.1519566, 20.1997890, -44.2444000, 45.4446030
1: -17.4038925, 17.3661079, -20.3969364, 20.2608223, -37.6647072, 37.7630463
2: -24.4552078, 16.6612263, -28.6462936, 19.4487343, -43.9039421, 45.3075142
3: -28.1345348, 13.9467545, -32.9332085, 16.2559814, -44.3905182, 46.8799591
4: -28.8498211, 16.6751919, -33.7843437, 19.4618378, -48.3116493, 50.4595337
5: -25.3716850, 14.8603659, -29.7260952, 17.3403358, -42.7120132, 44.5864601
6: -28.6497498, 14.1783848, -33.5008812, 16.5191956, -45.1689415, 47.6792641
7: -21.5967255, 21.1673965, -25.2642784, 24.7388916, -46.3356171, 46.4316750
8: -31.2083263, 15.3605251, -36.5813560, 17.9086685, -49.1169891, 51.9418716
9: -20.6089249, 20.8881912, -24.0797215, 24.4338264, -45.0427513, 44.9679108

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4885683, upper bound: 36.4884292
time: 8.59 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4881543, upper bound: 36.4880603
time: 8.38 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -24.5948181, 17.6807594, -29.1561852, 20.9273281, -45.5221405, 46.8369446
1: -17.8016930, 17.7559242, -21.1388264, 20.9722481, -38.7739410, 38.8947525
2: -25.0200672, 17.0284824, -29.6841640, 20.1343803, -45.1544495, 46.7126465
3: -28.7732391, 14.2579107, -34.1131134, 16.8291321, -45.6023674, 48.3710251
4: -29.5109024, 17.0417309, -34.9726868, 20.1618385, -49.6727409, 52.0144081
5: -25.9522381, 15.1954536, -30.7709999, 17.9663315, -43.9185600, 45.9664536
6: -29.3180466, 14.4874344, -34.6548462, 17.1508999, -46.4689484, 49.1422729
7: -22.0898743, 21.6473007, -26.1753464, 25.6099129, -47.6997871, 47.8226471
8: -31.9238148, 15.6971397, -37.8686829, 18.5657673, -50.4895821, 53.5658188
9: -21.0694199, 21.3589535, -24.9421883, 25.3094444, -46.3788643, 46.3011398

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4911873, upper bound: 36.4910600
time: 12.64 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4906134, upper bound: 36.4905642
time: 9.29 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -24.2570286, 17.4357929, -23.9625874, 17.2137794, -41.4708099, 41.3983803
1: -17.5576477, 17.5205231, -17.3306255, 17.3089485, -34.8665962, 34.8511505
2: -24.6772308, 16.7890053, -24.3679619, 16.5822678, -41.2594872, 41.1569672
3: -28.3915768, 14.0675278, -28.0322762, 13.8964930, -42.2880707, 42.0998039
4: -29.1256218, 16.8121643, -28.7780685, 16.5895100, -45.7151337, 45.5902252
5: -25.6181297, 14.9793682, -25.3127193, 14.7936592, -40.4117889, 40.2920876
6: -28.9411354, 14.2613144, -28.6229649, 14.0529518, -42.9940872, 42.8842773
7: -21.7880440, 21.3623543, -21.5144711, 21.1060066, -42.8940430, 42.8768234
8: -31.4882870, 15.4741745, -31.1165848, 15.2645206, -46.7528076, 46.5907593
9: -20.7896881, 21.0726089, -20.5240574, 20.8054008, -41.5950890, 41.5966644

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4807483, upper bound: 36.4802242
time: 8.88 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4795947, upper bound: 36.4792101
time: 42.04 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -24.7815018, 17.8076363, -25.0168190, 17.9775162, -42.7590179, 42.8244553
1: -17.9380836, 17.8934307, -18.1101990, 18.0576000, -35.9956818, 36.0036316
2: -25.2175484, 17.1432095, -25.4584503, 17.3027172, -42.5202637, 42.6016541
3: -29.0004272, 14.3656082, -29.2719555, 14.4983206, -43.4987411, 43.6375656
4: -29.7510414, 17.1659870, -30.0289345, 17.3250713, -47.0761108, 47.1949196
5: -26.1689911, 15.3029976, -26.4154358, 15.4480162, -41.6170082, 41.7184334
6: -29.5748634, 14.5665560, -29.8455982, 14.7108555, -44.2857056, 44.4121475
7: -22.2597160, 21.8200970, -22.4708881, 22.0230980, -44.2828140, 44.2909775
8: -32.1694412, 15.7999144, -32.4735870, 15.9503422, -48.1197815, 48.2734985
9: -21.2302456, 21.5228863, -21.4290009, 21.7254868, -42.9557343, 42.9518852

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4812831, upper bound: 36.4807650
time: 9.35 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4801374, upper bound: 36.4797520
time: 9.59 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -23.7625465, 17.0797577, -29.4591522, 21.1451054, -44.9076538, 46.5389099
1: -17.1919270, 17.1691113, -21.3521900, 21.1758041, -38.3677292, 38.5213013
2: -24.1635551, 16.4543343, -30.0122776, 20.3477402, -44.5112953, 46.4666100
3: -27.8054085, 13.7843685, -34.4441605, 16.9895878, -44.7949982, 48.2285309
4: -28.5370731, 16.4640923, -35.3232613, 20.3540154, -48.8910904, 51.7873535
5: -25.1010857, 14.6745796, -31.0739613, 18.1495533, -43.2506409, 45.7485352
6: -28.3692436, 13.9542656, -34.9713135, 17.3500195, -45.7192535, 48.9255791
7: -21.3383598, 20.9328747, -26.4523125, 25.8624420, -47.2008018, 47.3851814
8: -30.8535538, 15.1535206, -38.2523232, 18.7594204, -49.6129761, 53.4058456
9: -20.3624172, 20.6385498, -25.1979866, 25.5561790, -45.9185944, 45.8365288

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4891409, upper bound: 36.4890100
time: 8.96 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4887414, upper bound: 36.4886638
time: 9.19 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -24.2791500, 17.4452534, -30.5116634, 21.9083328, -46.1874809, 47.9569092
1: -17.5656853, 17.5359516, -22.1282616, 21.9205418, -39.4862289, 39.6642151
2: -24.6954575, 16.8021965, -31.1035633, 21.0667000, -45.7621536, 47.9057541
3: -28.4046211, 14.0776548, -35.6775551, 17.5880699, -45.9926872, 49.7552109
4: -29.1531067, 16.8109932, -36.5657196, 21.0861855, -50.2392883, 53.3767014
5: -25.6431732, 14.9930735, -32.1649666, 18.8070831, -44.4502563, 47.1580391
6: -28.9923592, 14.2550869, -36.1711159, 18.0192604, -47.0116196, 50.4262009
7: -21.8027534, 21.3825493, -27.4088440, 26.7720757, -48.5748291, 48.7913933
8: -31.5236015, 15.4738331, -39.5983124, 19.4494247, -50.9730110, 55.0721436
9: -20.7959061, 21.0815239, -26.1029968, 26.4713440, -47.2672501, 47.1845207

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4957907, upper bound: 36.4958755
time: 8.60 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4957907, upper bound: 36.4964495
time: 13.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.06 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 23.06
Output dim: 6, lower bound: -36.4785921, upper bound: 36.4781531
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 23.06
Output dim: 6, lower bound: -36.4775561, upper bound: 36.4772231
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 23.06
Output dim: 6, lower bound: -36.4791285, upper bound: 36.4786558
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 23.06
Output dim: 6, lower bound: -36.4780822, upper bound: 36.4777395
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 23.06
Output dim: 6, lower bound: -36.4885683, upper bound: 36.4884292
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 23.06
Output dim: 6, lower bound: -36.4881543, upper bound: 36.4880603
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.06
Output dim: 6, lower bound: -36.4911873, upper bound: 36.4910600
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.06
Output dim: 6, lower bound: -36.4906134, upper bound: 36.4905642
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 23.06
Output dim: 6, lower bound: -36.4807483, upper bound: 36.4802242
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 23.06
Output dim: 6, lower bound: -36.4795947, upper bound: 36.4792101
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 23.06
Output dim: 6, lower bound: -36.4812831, upper bound: 36.4807650
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 23.06
Output dim: 6, lower bound: -36.4801374, upper bound: 36.4797520
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 23.06
Output dim: 6, lower bound: -36.4891409, upper bound: 36.4890100
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 23.06
Output dim: 6, lower bound: -36.4887414, upper bound: 36.4886638
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.06
Output dim: 6, lower bound: -36.4957907, upper bound: 36.4958755
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.06
Output dim: 6, lower bound: -36.4957907, upper bound: 36.4964495

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -24.0813599, 17.3038044, -27.9815750, 20.0694275, -44.1507759, 45.2853775
1: -17.4164543, 17.3872814, -20.2588539, 20.1327419, -37.5491943, 37.6461334
2: -24.4831810, 16.6728458, -28.4593563, 19.3276711, -43.8108521, 45.1321983
3: -28.1632881, 13.9611807, -32.7218475, 16.1509609, -44.3142471, 46.6830254
4: -28.9047966, 16.6710110, -33.5903854, 19.3173332, -48.2221298, 50.2613983
5: -25.4182434, 14.8712769, -29.5572586, 17.2227669, -42.6410065, 44.4285355
6: -28.7334194, 14.1523161, -33.3348732, 16.3819256, -45.1153450, 47.4871902
7: -21.6190071, 21.1993656, -25.0996952, 24.5911026, -46.2101097, 46.2990494
8: -31.2647076, 15.3529835, -36.3690262, 17.7765656, -49.0412674, 51.7220001
9: -20.6216450, 20.9058418, -23.9204063, 24.2730579, -44.8947029, 44.8262482

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4728539, upper bound: 36.4728623
time: 10.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4706656, upper bound: 36.4705234
time: 14.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -23.2534447, 16.6864471, -33.2870865, 23.8327694, -47.0862122, 49.9735298
1: -16.7876663, 16.7876148, -24.1412392, 23.8913574, -40.6790161, 40.9288559
2: -23.6126728, 16.0922966, -33.9297485, 22.8865089, -46.4991837, 50.0220451
3: -27.1806946, 13.4756966, -38.9980888, 19.1401672, -46.3208542, 52.4737854
4: -27.9395714, 16.0588455, -40.0848770, 22.8944626, -50.8340302, 56.1437225
5: -24.5663891, 14.3335285, -35.2736893, 20.3680134, -44.9344025, 49.6072121
6: -27.8022099, 13.5857201, -39.6977997, 19.3359222, -47.1381302, 53.2835121
7: -20.8540421, 20.4786816, -29.8901405, 29.2406979, -50.0947189, 50.3688202
8: -30.2033653, 14.7818279, -43.3488693, 21.0065269, -51.2098808, 58.1306992
9: -19.8966331, 20.1682453, -28.4430923, 28.8916531, -48.7882805, 48.6113358

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4721114, upper bound: 36.4722310
time: 8.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4700599, upper bound: 36.4699842
time: 18.01 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -24.2791500, 17.4452534, -30.4700508, 21.8840485, -46.1631927, 47.9152985
1: -17.5656853, 17.5359516, -22.0985947, 21.8900909, -39.4557724, 39.6345444
2: -24.6954575, 16.8021965, -31.0590363, 21.0540428, -45.7494926, 47.8612289
3: -28.4046211, 14.0776548, -35.6310310, 17.5632248, -45.9678459, 49.7086868
4: -29.1531067, 16.8109932, -36.5049324, 21.0686226, -50.2217255, 53.3159256
5: -25.6431732, 14.9930735, -32.1024399, 18.7852859, -44.4284592, 47.0955124
6: -28.9923592, 14.2550869, -36.0947762, 18.0228176, -47.0151749, 50.3498611
7: -21.8027534, 21.3825493, -27.3725128, 26.7285995, -48.5313530, 48.7550583
8: -31.5236015, 15.4738331, -39.5423279, 19.4331818, -50.9567795, 55.0161591
9: -20.7959061, 21.0815239, -26.0691109, 26.4377975, -47.2337036, 47.1506348

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4779710, upper bound: 36.4782425
time: 9.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4765963, upper bound: 36.4766564
time: 7.49 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -24.2791500, 17.4452534, -30.1301460, 21.6331329, -45.9122696, 47.5753937
1: -17.5656853, 17.5359516, -21.8470230, 21.6508560, -39.2165413, 39.3829689
2: -24.6954575, 16.8021965, -30.7086411, 20.8083191, -45.5037689, 47.5108337
3: -28.4046211, 14.0776548, -35.2299652, 17.3719883, -45.7766113, 49.3076210
4: -29.1531067, 16.8109932, -36.1124878, 20.8223534, -49.9754601, 52.9234810
5: -25.6431732, 14.9930735, -31.7665939, 18.5710526, -44.2142258, 46.7596664
6: -28.9923592, 14.2550869, -35.7347412, 17.7829056, -46.7752647, 49.9898300
7: -21.8027534, 21.3825493, -27.0635891, 26.4420128, -48.2447662, 48.4461365
8: -31.5236015, 15.4738331, -39.1075745, 19.2026196, -50.7262115, 54.5814056
9: -20.7959061, 21.0815239, -25.7756653, 26.1401100, -46.9360123, 46.8571892

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 64

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4779710, upper bound: 36.4803774
time: 11.15 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4765963, upper bound: 36.4785205
time: 78.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 91.18 seconds
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 91.18
Output dim: 6, lower bound: -36.4728539, upper bound: 36.4728623
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 91.18
Output dim: 6, lower bound: -36.4706656, upper bound: 36.4705234
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 91.18
Output dim: 6, lower bound: -36.4721114, upper bound: 36.4722310
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 91.18
Output dim: 6, lower bound: -36.4700599, upper bound: 36.4699842
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 91.18
Output dim: 6, lower bound: -36.4779710, upper bound: 36.4782425
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 91.18
Output dim: 6, lower bound: -36.4765963, upper bound: 36.4766564
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 91.18
Output dim: 6, lower bound: -36.4779710, upper bound: 36.4803774
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 91.18
Output dim: 6, lower bound: -36.4765963, upper bound: 36.4785205
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=47.04602813720703
rel_dist={6: [-36.503033734627465, 36.50303373391884]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2375.45 seconds
