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
execution time: IAR + LP analysis = 1.05 + 8.95 = 10.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -36.5032747, upper bound: 36.5032750


# Binary Search by BASE starts (time budget: 2689.99 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=47.04602813720703
rel_dist={6: [-36.503194103113984, 36.50319409863076]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=47.04602813720703
rel_dist={6: [-36.50308451104452, 36.50308451104452]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=47.04602813720703
rel_dist={6: [-36.50297363759564, 36.50297363759566]}

## Binary Search Result
Binary search time: 36.49 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2653.51 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4706006, upper bound: 36.4707343
time: 42.00 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4583722, upper bound: 36.4583722
time: 6.46 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 48.57 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 48.57
Output dim: 6, lower bound: -36.4706006, upper bound: 36.4707343
IS_A2, status: Status.VERIFIED, split count: 1, time: 48.57
Output dim: 6, lower bound: -36.4583722, upper bound: 36.4583722
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=47.04602813720703
rel_dist={6: [-36.503194103113984, 36.50319409863076]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5031117, upper bound: 36.5029905
time: 7.66 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5032366, upper bound: 36.5032366
time: 6.25 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.03 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.03
Output dim: 6, lower bound: -36.5031117, upper bound: 36.5029905
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.03
Output dim: 6, lower bound: -36.5032366, upper bound: 36.5032366

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -26.1358681, 18.7964439, -26.3453789, 18.9361000, -45.0719681, 45.1418152
1: -18.9409447, 18.8499107, -19.0910988, 18.9999657, -37.9409103, 37.9410095
2: -26.6191959, 18.0804043, -26.8359642, 18.2049675, -44.8241653, 44.9163628
3: -30.5916786, 15.1372766, -30.8387356, 15.2553568, -45.8470345, 45.9760132
4: -31.3408966, 18.1206932, -31.6103420, 18.2510452, -49.5919418, 49.7310333
5: -27.5593796, 16.1499977, -27.8028259, 16.2686844, -43.8280602, 43.9528198
6: -31.0921783, 15.4537849, -31.3732872, 15.5381918, -46.6303711, 46.8270683
7: -23.4900723, 22.9867249, -23.6766434, 23.1772232, -46.6672974, 46.6633682
8: -33.9014206, 16.6999149, -34.1792717, 16.8135185, -50.7149353, 50.8791809
9: -22.3984489, 22.7051811, -22.5729752, 22.8854427, -45.2838898, 45.2781563

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4735034, upper bound: 36.4735447
time: 16.03 seconds

## Relational analysis of IS_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4942630, upper bound: 36.4936782
time: 7.64 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4896959, upper bound: 36.4894014
time: 323.34 seconds

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
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4749108, upper bound: 36.4753677
time: 11.86 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4583889, upper bound: 36.4583889
time: 5.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 18.54 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 18.54
Output dim: 6, lower bound: -36.4942630, upper bound: 36.4936782
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 18.54
Output dim: 6, lower bound: -36.4896959, upper bound: 36.4894014
IS_A2_A1, status: Status.VERIFIED, split count: 2, time: 18.54
Output dim: 6, lower bound: -36.4749108, upper bound: 36.4753677
IS_A2_A2, status: Status.VERIFIED, split count: 2, time: 18.54
Output dim: 6, lower bound: -36.4583889, upper bound: 36.4583889

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -26.1358681, 18.7964439, -25.1443062, 18.0673103, -44.2031708, 43.9407463
1: -18.9409447, 18.8499107, -18.1957512, 18.1522560, -37.0932007, 37.0456581
2: -26.6191959, 18.0804043, -25.5945225, 17.3747120, -43.9939079, 43.6749268
3: -30.5916786, 15.1372766, -29.4371223, 14.5699625, -45.1616402, 44.5743980
4: -31.3408966, 18.1206932, -30.2105446, 17.4034195, -48.7443123, 48.3312378
5: -27.5593796, 16.1499977, -26.5726585, 15.5094681, -43.0688477, 42.7226562
6: -31.0921783, 15.4537849, -30.0434341, 14.7537613, -45.8459396, 45.4972191
7: -23.4900723, 22.9867249, -22.5892754, 22.1431084, -45.6331787, 45.5760002
8: -33.9014206, 16.6999149, -32.6542892, 16.0202370, -49.9216576, 49.3541946
9: -22.3984489, 22.7051811, -21.5340080, 21.8366623, -44.2351112, 44.2391853

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4734861, upper bound: 36.4727481
time: 18.80 seconds

## Relational analysis of IS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4751044, upper bound: 36.4750578
time: 10.57 seconds

## Relational analysis of IS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4882206, upper bound: 36.4882973
time: 6.00 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4841719, upper bound: 36.4832556
time: 6.66 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 55.28 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 55.28
Output dim: 6, lower bound: -36.4882206, upper bound: 36.4882973
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 55.28
Output dim: 6, lower bound: -36.4841719, upper bound: 36.4832556
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=47.04602813720703
rel_dist={6: [-36.50323657019504, 36.50323656135174]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5031564, upper bound: 36.5030279
time: 5.19 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5032622, upper bound: 36.5032622
time: 4.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.15 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.15
Output dim: 6, lower bound: -36.5031564, upper bound: 36.5030279
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.15
Output dim: 6, lower bound: -36.5032622, upper bound: 36.5032622

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -26.1358681, 18.7964439, -26.4216805, 18.9908962, -45.1267548, 45.2181244
1: -18.9409447, 18.8499107, -19.1472130, 19.0538960, -37.9948387, 37.9971237
2: -26.6191959, 18.0804043, -26.9148560, 18.2566853, -44.8758812, 44.9952545
3: -30.5916786, 15.1372766, -30.9281616, 15.2985239, -45.8902016, 46.0654335
4: -31.3408966, 18.1206932, -31.7014198, 18.3034821, -49.6443710, 49.8221130
5: -27.5593796, 16.1499977, -27.8828411, 16.3154755, -43.8748550, 44.0328369
6: -31.0921783, 15.4537849, -31.4614029, 15.5846272, -46.6768036, 46.9151840
7: -23.4900723, 22.9867249, -23.7456284, 23.2433929, -46.7334595, 46.7323532
8: -33.9014206, 16.6999149, -34.2775040, 16.8623047, -50.7637253, 50.9774170
9: -22.3984489, 22.7051811, -22.6382275, 22.9516029, -45.3500519, 45.3433990

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4771190, upper bound: 36.4774519
time: 56.21 seconds

## Relational analysis of IS_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4950789, upper bound: 36.4944773
time: 8.99 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4897859, upper bound: 36.4894618
time: 5.09 seconds

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

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4774789, upper bound: 36.4780763
time: 8.29 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4583999, upper bound: 36.4583999
time: 7.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.61 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.61
Output dim: 6, lower bound: -36.4950789, upper bound: 36.4944773
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 16.61
Output dim: 6, lower bound: -36.4897859, upper bound: 36.4894618
IS_A2_A1, status: Status.VERIFIED, split count: 2, time: 16.61
Output dim: 6, lower bound: -36.4774789, upper bound: 36.4780763
IS_A2_A2, status: Status.VERIFIED, split count: 2, time: 16.61
Output dim: 6, lower bound: -36.4583999, upper bound: 36.4583999

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -26.1358681, 18.7964439, -25.2188759, 18.1207771, -44.2566376, 44.0153198
1: -18.9409447, 18.8499107, -18.2506924, 18.2048798, -37.1458206, 37.1006012
2: -26.6191959, 18.0804043, -25.6715126, 17.4251823, -44.0443802, 43.7519112
3: -30.5916786, 15.1372766, -29.5245209, 14.6122541, -45.2039299, 44.6617966
4: -31.3408966, 18.1206932, -30.2997246, 17.4546928, -48.7955894, 48.4204178
5: -27.5593796, 16.1499977, -26.6511688, 15.5550051, -43.1143837, 42.8011665
6: -31.0921783, 15.4537849, -30.1302528, 14.7985592, -45.8907394, 45.5840340
7: -23.4900723, 22.9867249, -22.6566315, 22.2079926, -45.6980629, 45.6433563
8: -33.9014206, 16.6999149, -32.7506714, 16.0677357, -49.9691544, 49.4505844
9: -22.3984489, 22.7051811, -21.5975780, 21.9014721, -44.2999191, 44.3027534

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4659251, upper bound: 36.4651063
time: 6.96 seconds

## Relational analysis of IS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4795658, upper bound: 36.4790801
time: 8.66 seconds

## Relational analysis of IS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4809881, upper bound: 36.4809989
time: 9.38 seconds

## Relational analysis of IS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4917245, upper bound: 36.4911470
time: 7.71 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4918049, upper bound: 36.4911392
time: 16.85 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 62.02 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 62.02
Output dim: 6, lower bound: -36.4917245, upper bound: 36.4911470
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 62.02
Output dim: 6, lower bound: -36.4918049, upper bound: 36.4911392

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -21.6297932, 15.4696941, -25.2188759, 18.1207771, -39.7505646, 40.6885681
1: -15.6008205, 15.6760015, -18.2506924, 18.2048798, -33.8056946, 33.9266930
2: -21.9610901, 14.9812698, -25.6715126, 17.4251823, -39.3862686, 40.6527824
3: -25.3259010, 12.5627289, -29.5245209, 14.6122541, -39.9381561, 42.0872498
4: -26.1008530, 14.9273767, -30.2997246, 17.4546928, -43.5555382, 45.2270966
5: -22.8612709, 13.3061333, -26.6511688, 15.5550051, -38.4162750, 39.9573021
6: -25.9593468, 12.4771805, -30.1302528, 14.7985592, -40.7579041, 42.6074257
7: -19.4225292, 19.0889893, -22.6566315, 22.2079926, -41.6305237, 41.7456207
8: -28.1246815, 13.7087355, -32.7506714, 16.0677357, -44.1924171, 46.4594078
9: -18.5153503, 18.7825680, -21.5975780, 21.9014721, -40.4168243, 40.3801384

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 188

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4623088, upper bound: 36.4612869
time: 6.08 seconds

## Relational analysis of IS_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4757782, upper bound: 36.4754708
time: 16.36 seconds

## Relational analysis of IS_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4846530, upper bound: 36.4830988
time: 10.45 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4813644, upper bound: 36.4803632
time: 5.59 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -24.0264034, 17.1862030, -25.2188759, 18.1207771, -42.1471786, 42.4050789
1: -17.3654785, 17.3807564, -18.2506924, 18.2048798, -35.5703506, 35.6314468
2: -24.4381523, 16.6011219, -25.6715126, 17.4251823, -41.8633270, 42.2726364
3: -28.1545238, 13.9254932, -29.5245209, 14.6122541, -42.7667770, 43.4500122
4: -29.0014629, 16.5746593, -30.2997246, 17.4546928, -46.4561539, 46.8743820
5: -25.4137077, 14.7520409, -26.6511688, 15.5550051, -40.9687042, 41.4032097
6: -28.7921162, 13.8810940, -30.1302528, 14.7985592, -43.5906754, 44.0113449
7: -21.5930653, 21.1870461, -22.6566315, 22.2079926, -43.8010559, 43.8436775
8: -31.2554989, 15.2052402, -32.7506714, 16.0677357, -47.3232307, 47.9559097
9: -20.5674591, 20.8687916, -21.5975780, 21.9014721, -42.4689255, 42.4663582

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 188

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4624589, upper bound: 36.4613662
time: 8.23 seconds

## Relational analysis of IS_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4758622, upper bound: 36.4754557
time: 6.50 seconds

## Relational analysis of IS_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4856291, upper bound: 36.4841118
time: 20.11 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4823939, upper bound: 36.4813991
time: 7.96 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 53.47 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 53.47
Output dim: 6, lower bound: -36.4846530, upper bound: 36.4830988
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 53.47
Output dim: 6, lower bound: -36.4813644, upper bound: 36.4803632
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 53.47
Output dim: 6, lower bound: -36.4856291, upper bound: 36.4841118
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 53.47
Output dim: 6, lower bound: -36.4823939, upper bound: 36.4813991
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=47.04602813720703
rel_dist={6: [-36.503262448646396, 36.50326217370076]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5031756, upper bound: 36.5030451
time: 4.45 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.5032747, upper bound: 36.5032749
time: 5.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.98 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.98
Output dim: 6, lower bound: -36.5031756, upper bound: 36.5030451
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.98
Output dim: 6, lower bound: -36.5032747, upper bound: 36.5032749

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -26.1358681, 18.7964439, -26.4216805, 18.9908962, -45.1267548, 45.2181244
1: -18.9409447, 18.8499107, -19.1472130, 19.0538960, -37.9948387, 37.9971237
2: -26.6191959, 18.0804043, -26.9148560, 18.2566853, -44.8758812, 44.9952545
3: -30.5916786, 15.1372766, -30.9281616, 15.2985239, -45.8902016, 46.0654335
4: -31.3408966, 18.1206932, -31.7014198, 18.3034821, -49.6443710, 49.8221130
5: -27.5593796, 16.1499977, -27.8828411, 16.3154755, -43.8748550, 44.0328369
6: -31.0921783, 15.4537849, -31.4614029, 15.5846272, -46.6768036, 46.9151840
7: -23.4900723, 22.9867249, -23.7456284, 23.2433929, -46.7334595, 46.7323532
8: -33.9014206, 16.6999149, -34.2775040, 16.8623047, -50.7637253, 50.9774170
9: -22.3984489, 22.7051811, -22.6382275, 22.9516029, -45.3500519, 45.3433990

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4785112, upper bound: 36.4788965
time: 5.86 seconds

## Relational analysis of IS_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4954431, upper bound: 36.4948495
time: 5.68 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4898254, upper bound: 36.4894888
time: 4.50 seconds

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

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4786966, upper bound: 36.4792819
time: 4.33 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4584054, upper bound: 36.4584054
time: 5.41 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 10.84 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 10.84
Output dim: 6, lower bound: -36.4954431, upper bound: 36.4948495
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 10.84
Output dim: 6, lower bound: -36.4898254, upper bound: 36.4894888
IS_A2_A1, status: Status.VERIFIED, split count: 2, time: 10.84
Output dim: 6, lower bound: -36.4786966, upper bound: 36.4792819
IS_A2_A2, status: Status.VERIFIED, split count: 2, time: 10.84
Output dim: 6, lower bound: -36.4584054, upper bound: 36.4584054

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -26.1358681, 18.7964439, -25.2188759, 18.1207771, -44.2566376, 44.0153198
1: -18.9409447, 18.8499107, -18.2506924, 18.2048798, -37.1458206, 37.1006012
2: -26.6191959, 18.0804043, -25.6715126, 17.4251823, -44.0443802, 43.7519112
3: -30.5916786, 15.1372766, -29.5245209, 14.6122541, -45.2039299, 44.6617966
4: -31.3408966, 18.1206932, -30.2997246, 17.4546928, -48.7955894, 48.4204178
5: -27.5593796, 16.1499977, -26.6511688, 15.5550051, -43.1143837, 42.8011665
6: -31.0921783, 15.4537849, -30.1302528, 14.7985592, -45.8907394, 45.5840340
7: -23.4900723, 22.9867249, -22.6566315, 22.2079926, -45.6980629, 45.6433563
8: -33.9014206, 16.6999149, -32.7506714, 16.0677357, -49.9691544, 49.4505844
9: -22.3984489, 22.7051811, -21.5975780, 21.9014721, -44.2999191, 44.3027534

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4694344, upper bound: 36.4691457
time: 5.02 seconds

## Relational analysis of IS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4815028, upper bound: 36.4809130
time: 5.78 seconds

## Relational analysis of IS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4921014, upper bound: 36.4915140
time: 7.51 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -36.4921787, upper bound: 36.4915082
time: 8.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 35.50 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 35.50
Output dim: 6, lower bound: -36.4921014, upper bound: 36.4915140
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 35.50
Output dim: 6, lower bound: -36.4921787, upper bound: 36.4915082

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -21.6297932, 15.4696941, -25.2188759, 18.1207771, -39.7505646, 40.6885681
1: -15.6008205, 15.6760015, -18.2506924, 18.2048798, -33.8056946, 33.9266930
2: -21.9610901, 14.9812698, -25.6715126, 17.4251823, -39.3862686, 40.6527824
3: -25.3259010, 12.5627289, -29.5245209, 14.6122541, -39.9381561, 42.0872498
4: -26.1008530, 14.9273767, -30.2997246, 17.4546928, -43.5555382, 45.2270966
5: -22.8612709, 13.3061333, -26.6511688, 15.5550051, -38.4162750, 39.9573021
6: -25.9593468, 12.4771805, -30.1302528, 14.7985592, -40.7579041, 42.6074257
7: -19.4225292, 19.0889893, -22.6566315, 22.2079926, -41.6305237, 41.7456207
8: -28.1246815, 13.7087355, -32.7506714, 16.0677357, -44.1924171, 46.4594078
9: -18.5153503, 18.7825680, -21.5975780, 21.9014721, -40.4168243, 40.3801384

Time for backsubstitution: 0.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 188

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4658454, upper bound: 36.4653245
time: 5.16 seconds

## Relational analysis of IS_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4776985, upper bound: 36.4773266
time: 5.98 seconds

## Relational analysis of IS_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4853682, upper bound: 36.4837499
time: 4.39 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4818637, upper bound: 36.4808185
time: 5.31 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -24.0264034, 17.1862030, -25.2188759, 18.1207771, -42.1471786, 42.4050789
1: -17.3654785, 17.3807564, -18.2506924, 18.2048798, -35.5703506, 35.6314468
2: -24.4381523, 16.6011219, -25.6715126, 17.4251823, -41.8633270, 42.2726364
3: -28.1545238, 13.9254932, -29.5245209, 14.6122541, -42.7667770, 43.4500122
4: -29.0014629, 16.5746593, -30.2997246, 17.4546928, -46.4561539, 46.8743820
5: -25.4137077, 14.7520409, -26.6511688, 15.5550051, -40.9687042, 41.4032097
6: -28.7921162, 13.8810940, -30.1302528, 14.7985592, -43.5906754, 44.0113449
7: -21.5930653, 21.1870461, -22.6566315, 22.2079926, -43.8010559, 43.8436775
8: -31.2554989, 15.2052402, -32.7506714, 16.0677357, -47.3232307, 47.9559097
9: -20.5674591, 20.8687916, -21.5975780, 21.9014721, -42.4689255, 42.4663582

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 201
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 201
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 188

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4659944, upper bound: 36.4653752
time: 5.06 seconds

## Relational analysis of IS_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4777808, upper bound: 36.4773204
time: 4.75 seconds

## Relational analysis of IS_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4863123, upper bound: 36.4847155
time: 6.96 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -36.4828510, upper bound: 36.4818247
time: 6.93 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 34.24 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 34.24
Output dim: 6, lower bound: -36.4853682, upper bound: 36.4837499
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 34.24
Output dim: 6, lower bound: -36.4818637, upper bound: 36.4808185
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 34.24
Output dim: 6, lower bound: -36.4863123, upper bound: 36.4847155
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 34.24
Output dim: 6, lower bound: -36.4828510, upper bound: 36.4818247
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=47.04602813720703
rel_dist={6: [-36.503274688156715, 36.50327495642699]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 943.86 seconds
