## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 318.144348814
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-206.9136353, 163.8058472, -206.9136353, 163.8058472, -370.7194519, 370.7194519)
1: (-174.4282379, 145.3141937, -174.4282379, 145.3141937, -319.7423401, 319.7423401)
2: (-227.8242950, 147.6721191, -227.8242950, 147.6721191, -375.4963989, 375.4963989)
3: (-242.0703125, 127.7758865, -242.0703125, 127.7758865, -369.8461914, 369.8461914)
4: (-222.2664337, 169.9702454, -222.2664337, 169.9702454, -392.2366943, 392.2366943)
5: (-198.4908142, 154.4979553, -198.4908142, 154.4979553, -352.9887390, 352.9887390)
6: (-190.0688934, 183.4619141, -190.0688934, 183.4619141, -373.5308228, 373.5308228)
7: (-207.8973694, 174.0781555, -207.8973694, 174.0781555, -381.9755249, 381.9755249)
8: (-250.0145874, 170.8148804, -250.0145874, 170.8148804, -420.8294678, 420.8294678)
9: (-188.8386993, 185.9809875, -188.8386993, 185.9809875, -374.8196716, 374.8196716)

## BASE Result
execution time: IAR + LP analysis = 1.24 + 10.86 = 12.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -318.2353807, upper bound: 318.2353807


# Binary Search by BASE starts (time budget: 2687.90 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=319.7423400878906
rel_dist={1: [-318.2353263992982, 318.2353263992983]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=319.7423400878906
rel_dist={1: [-318.23529533371016, 318.23529533356407]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=319.7423400878906
rel_dist={1: [-318.2352719030525, 318.23527182273534]}

## Binary Search Result
Binary search time: 42.79 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2645.11 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2058406, upper bound: 318.2023116
time: 10.66 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1926209, upper bound: 318.1926210
time: 6.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.12 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.12
Output dim: 1, lower bound: -318.2058406, upper bound: 318.2023116
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.12
Output dim: 1, lower bound: -318.1926209, upper bound: 318.1926210

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -205.5787811, 162.7531281, -206.9136353, 163.8058472, -369.3846436, 369.6666870
1: -173.3099365, 144.3747864, -174.4282379, 145.3141937, -318.6240234, 318.8029480
2: -226.3644714, 146.7181244, -227.8242950, 147.6721191, -374.0365906, 374.5424194
3: -240.5008850, 126.9480591, -242.0703125, 127.7758865, -368.2767639, 369.0183716
4: -220.8441620, 168.8690796, -222.2664337, 169.9702454, -390.8143921, 391.1354980
5: -197.2106323, 153.4936523, -198.4908142, 154.4979553, -351.7085876, 351.9844666
6: -188.8430939, 182.2836456, -190.0688934, 183.4619141, -372.3049927, 372.3525391
7: -206.5591736, 172.9551849, -207.8973694, 174.0781555, -380.6373291, 380.8525391
8: -248.4152527, 169.7273712, -250.0145874, 170.8148804, -419.2301331, 419.7419434
9: -187.6236420, 184.7870789, -188.8386993, 185.9809875, -373.6046143, 373.6257324

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1913494, upper bound: 318.1894912
time: 8.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1732244, upper bound: 318.1636529
time: 9.04 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -200.4977570, 158.7754822, -205.9505005, 163.0445862, -363.5423584, 364.7259827
1: -169.0623627, 140.8095551, -173.6223907, 144.6379395, -313.7002869, 314.4319458
2: -220.9532623, 143.0125580, -226.7713165, 146.9853058, -367.9384766, 369.7838440
3: -234.6011963, 123.7453156, -240.9389496, 127.1806946, -361.7818909, 364.6842651
4: -215.5938873, 164.5584106, -221.2377777, 169.1762085, -384.7700806, 385.7962036
5: -192.3430634, 149.4841919, -197.5652618, 153.7733765, -346.1164551, 347.0494385
6: -184.3029785, 177.8934937, -189.1841278, 182.6126709, -366.9156494, 367.0776367
7: -201.4940186, 168.6785126, -206.9330902, 173.2688446, -374.7628479, 375.6115723
8: -242.4742432, 165.5976562, -248.8604584, 170.0301056, -412.5043335, 414.4580994
9: -183.0123749, 180.2741089, -187.9611664, 185.1198730, -368.1321411, 368.2352905

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1558253, upper bound: 318.1629955
time: 7.22 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1466538, upper bound: 318.1466538
time: 6.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.87 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.87
Output dim: 1, lower bound: -318.1913494, upper bound: 318.1894912
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.87
Output dim: 1, lower bound: -318.1732244, upper bound: 318.1636529
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.87
Output dim: 1, lower bound: -318.1558253, upper bound: 318.1629955
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.87
Output dim: 1, lower bound: -318.1466538, upper bound: 318.1466538

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -205.5787811, 162.7531281, -201.7486572, 159.7551727, -365.3339539, 364.5017700
1: -173.3099365, 144.3747864, -170.0928955, 141.6844482, -314.9942932, 314.4676514
2: -226.3644714, 146.7181244, -222.1750488, 143.9926147, -370.3570862, 368.8931580
3: -240.5008850, 126.9480591, -235.9363251, 124.5760193, -365.0768738, 362.8843384
4: -220.8441620, 168.8690796, -216.7253571, 165.7321167, -386.5762634, 385.5944214
5: -197.2106323, 153.4936523, -193.5281372, 150.6325836, -347.8432007, 347.0217896
6: -188.8430939, 182.2836456, -185.3204346, 178.8934479, -367.7365417, 367.6040649
7: -206.5591736, 172.9551849, -202.6959381, 169.7283020, -376.2874451, 375.6511230
8: -248.4152527, 169.7273712, -243.8268127, 166.6226654, -415.0379028, 413.5541992
9: -187.6236420, 184.7870789, -184.1246033, 181.3619080, -368.9855347, 368.9116211

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1237164, upper bound: 318.1276418
time: 11.32 seconds

## Relational analysis of IS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1869142, upper bound: 318.1854833
time: 8.90 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1908775, upper bound: 318.1888794
time: 7.99 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -203.5606232, 161.1670074, -204.1518860, 161.7150726, -365.2756958, 365.3188782
1: -171.6150665, 142.9552612, -172.0771484, 143.3549652, -314.9700012, 315.0324097
2: -224.1569519, 145.2760315, -224.8767548, 145.6356659, -369.7926025, 370.1527710
3: -238.1044617, 125.6978683, -238.6658020, 125.9837494, -364.0881348, 364.3636169
4: -218.6732330, 167.2090759, -219.3159637, 167.6087952, -386.2820129, 386.5250244
5: -195.2695923, 151.9794312, -195.8265228, 152.3001862, -347.5697632, 347.8059692
6: -186.9846649, 180.4972534, -187.5478821, 181.0230103, -368.0076294, 368.0451355
7: -204.5244598, 171.2541046, -205.0476685, 171.7090454, -376.2334595, 376.3017578
8: -245.9953461, 168.0863037, -246.7667084, 168.6043701, -414.5997009, 414.8530273
9: -185.7766266, 182.9784851, -186.2518005, 183.5061493, -369.2827454, 369.2302856

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1685146, upper bound: 318.1598389
time: 8.78 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1727166, upper bound: 318.1633569
time: 8.83 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -200.4977570, 158.7754822, -200.7729187, 158.9840088, -359.4817505, 359.5483704
1: -169.0623627, 140.8095551, -169.2764282, 140.9990082, -310.0613708, 310.0859985
2: -220.9532623, 143.0125580, -221.1078796, 143.2960663, -364.2492981, 364.1203918
3: -234.6011963, 123.7453156, -234.7897491, 123.9725494, -358.5736694, 358.5350647
4: -215.5938873, 164.5584106, -215.6829376, 164.9276581, -380.5214539, 380.2413330
5: -192.3430634, 149.4841919, -192.5905151, 149.8983765, -342.2414551, 342.0746460
6: -184.3029785, 177.8934937, -184.4238739, 178.0325623, -362.3355408, 362.3173828
7: -201.4940186, 168.6785126, -201.7185211, 168.9078979, -370.4018555, 370.3970032
8: -242.4742432, 165.5976562, -242.6572418, 165.8271790, -408.3014221, 408.2548828
9: -183.0123749, 180.2741089, -183.2352142, 180.4890442, -363.5014038, 363.5093079

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1511925, upper bound: 318.1571398
time: 7.89 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1555035, upper bound: 318.1624887
time: 7.48 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -198.6006012, 157.2853088, -203.2323608, 160.9881897, -359.5888062, 360.5176392
1: -167.4696655, 139.4764557, -171.3076935, 142.7091980, -310.1788635, 310.7840881
2: -218.8789978, 141.6580963, -223.8712616, 144.9797363, -363.8587341, 365.5293579
3: -232.3509216, 122.5700302, -237.5861664, 125.4152069, -357.7661133, 360.1561890
4: -213.5545044, 163.0001831, -218.3342438, 166.8508759, -380.4053955, 381.3343811
5: -190.5188446, 148.0601349, -194.9429016, 151.6082306, -342.1270142, 343.0030212
6: -182.5574646, 176.2152252, -186.7037048, 180.2121277, -362.7695923, 362.9189453
7: -199.5823975, 167.0798492, -204.1272583, 170.9361420, -370.5185547, 371.2070923
8: -240.2008820, 164.0561371, -245.6649475, 167.8554230, -408.0562134, 409.7210388
9: -181.2771454, 178.5748444, -185.4141083, 182.6837463, -363.9608765, 363.9889221

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1361930, upper bound: 318.1352544
time: 8.16 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1384761, upper bound: 318.1384761
time: 7.06 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 49.10 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 49.10
Output dim: 1, lower bound: -318.1869142, upper bound: 318.1854833
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 49.10
Output dim: 1, lower bound: -318.1908775, upper bound: 318.1888794
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 49.10
Output dim: 1, lower bound: -318.1685146, upper bound: 318.1598389
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 49.10
Output dim: 1, lower bound: -318.1727166, upper bound: 318.1633569
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 49.10
Output dim: 1, lower bound: -318.1511925, upper bound: 318.1571398
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 49.10
Output dim: 1, lower bound: -318.1555035, upper bound: 318.1624887
IS_A2_B2_B1, status: Status.VERIFIED, split count: 3, time: 49.10
Output dim: 1, lower bound: -318.1361930, upper bound: 318.1352544
IS_A2_B2_B2, status: Status.VERIFIED, split count: 3, time: 49.10
Output dim: 1, lower bound: -318.1384761, upper bound: 318.1384761

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -200.2455444, 158.5408020, -200.8823700, 159.0706482, -359.3161926, 359.4231567
1: -168.8929291, 140.6300659, -169.3715668, 141.0759888, -309.9689026, 310.0016174
2: -220.4480896, 142.9484863, -221.2171783, 143.3778839, -363.8259583, 364.1656494
3: -234.3122864, 123.6799774, -234.9268646, 124.0434189, -358.3557129, 358.6068115
4: -215.1802673, 164.5164490, -215.7996521, 165.0228577, -380.2031250, 380.3161011
5: -192.1372528, 149.5405121, -192.7004547, 149.9900665, -342.1272888, 342.2409363
6: -183.9815674, 177.5668335, -184.5262756, 178.1259613, -362.1075439, 362.0931091
7: -201.1806488, 168.4609680, -201.8237762, 168.9985352, -370.1791687, 370.2847290
8: -242.0312195, 165.4669342, -242.7854614, 165.9222717, -407.9534607, 408.2523193
9: -182.7977600, 180.0509491, -183.3362122, 180.5874634, -363.3851929, 363.3871460

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1388883, upper bound: 318.1331944
time: 9.15 seconds

## Relational analysis of IS_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1534878, upper bound: 318.1543731
time: 8.93 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1869142, upper bound: 318.1854833
time: 8.51 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1869142, upper bound: 318.1854833
time: 7.91 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -202.6089020, 160.4048615, -201.7486572, 159.7551727, -362.3640747, 362.1535034
1: -170.8432465, 142.2868958, -170.0928955, 141.6844482, -312.5276794, 312.3797913
2: -223.0740509, 144.6114807, -222.1750488, 143.9926147, -367.0666504, 366.7864990
3: -237.0509186, 125.1239929, -235.9363251, 124.5760193, -361.6269226, 361.0602417
4: -217.6773987, 166.4368744, -216.7253571, 165.7321167, -383.4094543, 383.1622314
5: -194.3752289, 151.2899323, -193.5281372, 150.6325836, -345.0078125, 344.8180542
6: -186.1242065, 179.6540222, -185.3204346, 178.8934479, -365.0176392, 364.9744263
7: -203.5657959, 170.4470520, -202.6959381, 169.7283020, -373.2940674, 373.1430054
8: -244.8517151, 167.3414764, -243.8268127, 166.6226654, -411.4743652, 411.1682739
9: -184.9213562, 182.1353149, -184.1246033, 181.3619080, -366.2832642, 366.2598877

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1232262, upper bound: 318.1272313
time: 9.22 seconds

## Relational analysis of IS_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1488527, upper bound: 318.1412709
time: 9.27 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1301076, upper bound: 318.1259598
time: 9.03 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -198.2431335, 156.9672852, -203.2850342, 161.0301056, -359.2731628, 360.2523193
1: -167.2106323, 139.2216797, -171.3550262, 142.7462006, -309.9567871, 310.5766296
2: -218.2580109, 141.5174866, -223.9182739, 145.0204010, -363.2784119, 365.4357605
3: -231.9345398, 122.4389877, -237.6558990, 125.4503860, -357.3848572, 360.0948486
4: -213.0265656, 162.8699493, -218.3894501, 166.8991394, -379.9257202, 381.2593994
5: -190.2112732, 148.0377960, -194.9984131, 151.6572113, -341.8684692, 343.0361938
6: -182.1376801, 175.7941742, -186.7532196, 180.2548828, -362.3925171, 362.5473328
7: -199.1617737, 166.7727966, -204.1748505, 170.9788361, -370.1405640, 370.9476013
8: -239.6304016, 163.8393860, -245.7242126, 167.9035950, -407.5339966, 409.5635986
9: -180.9654694, 178.2564240, -185.4629669, 182.7309570, -363.6964111, 363.7193909

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1478047, upper bound: 318.1434707
time: 8.87 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1685146, upper bound: 318.1598389
time: 8.74 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -200.5950623, 158.8221588, -204.1518860, 161.7150726, -362.3101196, 362.9740601
1: -169.1517487, 140.8705139, -172.0771484, 143.3549652, -312.5066833, 312.9476624
2: -220.8711853, 143.1723480, -224.8767548, 145.6356659, -366.5068359, 368.0491028
3: -234.6594696, 123.8764420, -238.6658020, 125.9837494, -360.6431580, 362.5422363
4: -215.5110016, 164.7804108, -219.3159637, 167.6087952, -383.1197815, 384.0963745
5: -192.4381409, 149.7789459, -195.8265228, 152.3001862, -344.7383423, 345.6054688
6: -184.2696838, 177.8715057, -187.5478821, 181.0230103, -365.2926331, 365.4193726
7: -201.5354156, 168.7493896, -205.0476685, 171.7090454, -373.2444458, 373.7970581
8: -242.4368896, 165.7039642, -246.7667084, 168.6043701, -411.0412598, 412.4706726
9: -183.0781708, 180.3305511, -186.2518005, 183.5061493, -366.5843201, 366.5823364

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1291725, upper bound: 318.1269014
time: 10.08 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1578113, upper bound: 318.1517737
time: 9.18 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1727166, upper bound: 318.1633569
time: 10.21 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -199.6280060, 158.0883942, -195.4416046, 154.7730103, -354.4010010, 353.5299988
1: -168.3377533, 140.1990356, -164.8597412, 137.2561188, -305.5938416, 305.0587769
2: -219.9917145, 142.3954926, -215.1938019, 139.5265198, -359.5182495, 357.5892334
3: -233.5874634, 123.2105026, -228.6032257, 120.7052231, -354.2926941, 351.8136902
4: -214.6643219, 163.8462524, -210.0201416, 160.5762329, -375.2405396, 373.8663940
5: -191.5119629, 148.8391571, -187.5178680, 145.9463806, -337.4583435, 336.3570251
6: -183.5054779, 177.1229858, -179.5628662, 173.3167725, -356.8222046, 356.6858521
7: -200.6185303, 167.9460754, -196.3406830, 164.4151306, -365.0336609, 364.2867432
8: -241.4281616, 164.8941040, -236.2745972, 161.5670776, -402.9952393, 401.1687012
9: -182.2211456, 179.4963074, -178.4095917, 175.7524414, -357.9735718, 357.9057922

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1511924, upper bound: 318.1571398
time: 7.35 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1511924, upper bound: 318.1571398
time: 7.98 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -200.4977570, 158.7754822, -197.8053589, 156.6377716, -357.1355286, 356.5808105
1: -169.0623627, 140.8095551, -166.8110809, 138.9129333, -307.9752808, 307.6206055
2: -220.9532623, 143.0125580, -217.8198853, 141.1911011, -362.1443176, 360.8323975
3: -234.6011963, 123.7453156, -231.3429260, 122.1492310, -356.7503357, 355.0881958
4: -215.5938873, 164.5584106, -212.5186768, 162.4974823, -378.0913696, 377.0770264
5: -192.3430634, 149.4841919, -189.7574463, 147.6959381, -340.0389709, 339.2416382
6: -184.3029785, 177.8934937, -181.7071381, 175.4048615, -359.7078247, 359.6006470
7: -201.4940186, 168.6785126, -198.7268829, 166.4017181, -367.8956909, 367.4053955
8: -242.4742432, 165.5976562, -239.0958557, 163.4433899, -405.9176331, 404.6935120
9: -183.0123749, 180.2741089, -180.5349731, 177.8390350, -360.8514099, 360.8090820

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1456800, upper bound: 318.1516454
time: 9.12 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1473177, upper bound: 318.1545662
time: 8.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 48.11 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 48.11
Output dim: 1, lower bound: -318.1869142, upper bound: 318.1854833
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 48.11
Output dim: 1, lower bound: -318.1869142, upper bound: 318.1854833
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 48.11
Output dim: 1, lower bound: -318.1488527, upper bound: 318.1412709
IS_A1_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 48.11
Output dim: 1, lower bound: -318.1301076, upper bound: 318.1259598
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 48.11
Output dim: 1, lower bound: -318.1478047, upper bound: 318.1434707
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 48.11
Output dim: 1, lower bound: -318.1685146, upper bound: 318.1598389
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 48.11
Output dim: 1, lower bound: -318.1578113, upper bound: 318.1517737
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 48.11
Output dim: 1, lower bound: -318.1727166, upper bound: 318.1633569
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 48.11
Output dim: 1, lower bound: -318.1511924, upper bound: 318.1571398
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 48.11
Output dim: 1, lower bound: -318.1511924, upper bound: 318.1571398
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 48.11
Output dim: 1, lower bound: -318.1456800, upper bound: 318.1516454
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 48.11
Output dim: 1, lower bound: -318.1473177, upper bound: 318.1545662
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=319.7423400878906
rel_dist={1: [-318.2353263992982, 318.2353263992983]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1976854, upper bound: 318.1995534
time: 9.09 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1925052, upper bound: 318.1925052
time: 7.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.18 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 17.18
Output dim: 1, lower bound: -318.1976854, upper bound: 318.1995534
IS_B2, status: Status.UNKNOWN, split count: 1, time: 17.18
Output dim: 1, lower bound: -318.1925052, upper bound: 318.1925052

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -206.9136353, 163.8058472, -205.5787811, 162.7531281, -369.6666870, 369.3846436
1: -174.4282379, 145.3141937, -173.3099365, 144.3747864, -318.8029480, 318.6240234
2: -227.8242950, 147.6721191, -226.3644714, 146.7181244, -374.5424194, 374.0365906
3: -242.0703125, 127.7758865, -240.5008850, 126.9480591, -369.0183716, 368.2767639
4: -222.2664337, 169.9702454, -220.8441620, 168.8690796, -391.1354980, 390.8143921
5: -198.4908142, 154.4979553, -197.2106323, 153.4936523, -351.9844666, 351.7085876
6: -190.0688934, 183.4619141, -188.8430939, 182.2836456, -372.3525391, 372.3049927
7: -207.8973694, 174.0781555, -206.5591736, 172.9551849, -380.8525391, 380.6373291
8: -250.0145874, 170.8148804, -248.4152527, 169.7273712, -419.7419434, 419.2301331
9: -188.8386993, 185.9809875, -187.6236420, 184.7870789, -373.6257324, 373.6046143

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1502203, upper bound: 318.1499406
time: 9.05 seconds

## Relational analysis of IS_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1787408, upper bound: 318.1791259
time: 9.94 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1555794, upper bound: 318.1620371
time: 10.60 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -204.3916321, 161.8123474, -200.4977570, 158.7754822, -363.1671143, 362.3100586
1: -172.3177643, 143.5432587, -169.0623627, 140.8095551, -313.1273193, 312.6056213
2: -225.0668182, 145.8735504, -220.9532623, 143.0125580, -368.0793457, 366.8267822
3: -239.1076660, 126.2172165, -234.6011963, 123.7453156, -362.8529663, 360.8184204
4: -219.5724945, 167.8905792, -215.5938873, 164.5584106, -384.1309204, 383.4844055
5: -196.0671997, 152.6008759, -192.3430634, 149.4841919, -345.5513916, 344.9439392
6: -187.7515411, 181.2381439, -184.3029785, 177.8934937, -365.6450195, 365.5411072
7: -205.3717499, 171.9587708, -201.4940186, 168.6785126, -374.0502625, 373.4527893
8: -246.9920502, 168.7592773, -242.4742432, 165.5976562, -412.5897217, 411.2335205
9: -186.5405273, 183.7256927, -183.0123749, 180.2741089, -366.8146362, 366.7380676

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1555742, upper bound: 318.1513424
time: 8.02 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464445
time: 8.84 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 18.14 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 18.14
Output dim: 1, lower bound: -318.1787408, upper bound: 318.1791259
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 18.14
Output dim: 1, lower bound: -318.1555794, upper bound: 318.1620371
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 18.14
Output dim: 1, lower bound: -318.1555742, upper bound: 318.1513424
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 18.14
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464445

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -201.7486572, 159.7551727, -205.1775818, 162.4384766, -364.1871338, 364.9327087
1: -170.0928955, 141.6844482, -172.9731750, 144.0928802, -314.1857605, 314.6575928
2: -222.1750488, 143.9926147, -225.9257050, 146.4323578, -368.6074219, 369.9183350
3: -235.9363251, 124.5760193, -240.0242004, 126.6995697, -362.6358337, 364.6001892
4: -216.7253571, 165.7321167, -220.4138489, 168.5398712, -385.2651978, 386.1459656
5: -193.5281372, 150.6325836, -196.8251953, 153.1933594, -346.7214966, 347.4577637
6: -185.3204346, 178.8934479, -188.4742126, 181.9287109, -367.2490845, 367.3676453
7: -202.6959381, 169.7283020, -206.1551666, 172.6172180, -375.3131409, 375.8834839
8: -243.8268127, 166.6226654, -247.9347076, 169.4019012, -413.2286682, 414.5573730
9: -184.1246033, 181.3619080, -187.2573700, 184.4282227, -368.5527954, 368.6192627

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1555793, upper bound: 318.1620371
time: 9.76 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1555793, upper bound: 318.1620371
time: 11.19 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -204.1518860, 161.7150726, -200.4252625, 158.7035675, -362.8554382, 362.1403198
1: -172.0771484, 143.3549652, -168.9827271, 140.7503204, -312.8274536, 312.3377075
2: -224.8767548, 145.6356659, -220.7277222, 143.0363312, -367.9130554, 366.3634033
3: -238.6658020, 125.9837494, -234.3831787, 123.7560730, -362.4218750, 360.3669434
4: -219.3159637, 167.6087952, -215.3016357, 164.6306152, -383.9465942, 382.9103394
5: -195.8265228, 152.3001862, -192.2548676, 149.6274719, -345.4539795, 344.5550537
6: -187.5478821, 181.0230103, -184.0982056, 177.7225494, -365.2703857, 365.1211853
7: -205.0476685, 171.7090454, -201.3643494, 168.6116791, -373.6593628, 373.0733337
8: -246.7667084, 168.6043701, -242.2360840, 165.5365753, -412.3032837, 410.8404541
9: -186.2518005, 183.5061493, -182.9074707, 180.1699829, -366.4217834, 366.4136047

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1555794, upper bound: 318.1620371
time: 10.64 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1555793, upper bound: 318.1620371
time: 9.98 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -199.2002716, 157.7405243, -200.0855560, 158.4520416, -357.6523132, 357.8260803
1: -167.9598083, 139.8940277, -168.7163849, 140.5198975, -308.4797058, 308.6103821
2: -219.3876801, 142.1737061, -220.5025177, 142.7187347, -362.1063843, 362.6762085
3: -232.9416351, 122.9999542, -234.1116638, 123.4896927, -356.4313354, 357.1116333
4: -214.0026855, 163.6305237, -215.1516571, 164.2201843, -378.2228699, 378.7821655
5: -191.0789185, 148.7147217, -191.9468842, 149.1755371, -340.2544556, 340.6616211
6: -182.9782410, 176.6448517, -183.9238434, 177.5288849, -360.5071411, 360.5686951
7: -200.1430664, 167.5855560, -201.0788727, 168.3312073, -368.4741821, 368.6644287
8: -240.7716980, 164.5447388, -241.9802094, 165.2630615, -406.0346985, 406.5249634
9: -181.8015289, 179.0815887, -182.6360779, 179.9051361, -361.7065735, 361.7176208

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464445
time: 9.41 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464446
time: 8.29 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -201.7379150, 159.8072052, -195.6477356, 154.9661713, -356.7040710, 355.4549561
1: -170.0571136, 141.6597900, -164.9913788, 137.4012451, -307.4583130, 306.6511841
2: -222.2375641, 143.9138794, -215.6508484, 139.5492859, -361.7868652, 359.5647278
3: -235.8313141, 124.4910660, -228.8490295, 120.7406158, -356.5718994, 353.3400879
4: -216.7390289, 165.6190948, -210.3807220, 160.5755005, -377.3145142, 375.9998169
5: -193.5068970, 150.4838409, -187.6801147, 145.8437042, -339.3505859, 338.1639404
6: -185.3315887, 178.8947906, -179.8412781, 173.6031799, -358.9347534, 358.7360840
7: -202.6311646, 169.6802368, -196.6067505, 164.5911255, -367.2222900, 366.2869568
8: -243.8740234, 166.6385040, -236.6633606, 161.6568756, -405.5308838, 403.3018799
9: -184.0529327, 181.3471680, -178.5752258, 175.9305115, -359.9834290, 359.9223633

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464445
time: 7.39 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464445
time: 6.24 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 14.91 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 14.91
Output dim: 1, lower bound: -318.1555793, upper bound: 318.1620371
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 14.91
Output dim: 1, lower bound: -318.1555793, upper bound: 318.1620371
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 14.91
Output dim: 1, lower bound: -318.1555794, upper bound: 318.1620371
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 14.91
Output dim: 1, lower bound: -318.1555793, upper bound: 318.1620371
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 14.91
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464445
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 14.91
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464446
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 14.91
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464445
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 14.91
Output dim: 1, lower bound: -318.1464446, upper bound: 318.1464445

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -201.7486572, 159.7551727, -200.4309692, 158.7160034, -360.4645996, 360.1861572
1: -170.0928955, 141.6844482, -168.9890137, 140.7574310, -310.8503113, 310.6734314
2: -222.1750488, 143.9926147, -220.7341919, 143.0513000, -365.2263184, 364.7268066
3: -235.9363251, 124.5760193, -234.3875580, 123.7591095, -359.6953735, 358.9635620
4: -216.7253571, 165.7321167, -215.3217163, 164.6451569, -381.3705139, 381.0537720
5: -193.5281372, 150.6325836, -192.2644653, 149.6413422, -343.1694946, 342.8970337
6: -185.3204346, 178.8934479, -184.1105347, 177.7305756, -363.0509338, 363.0039673
7: -202.6959381, 169.7283020, -201.3753967, 168.6199493, -371.3158875, 371.1036987
8: -243.8268127, 166.6226654, -242.2480469, 165.5495605, -409.3763733, 408.8707275
9: -184.1246033, 181.3619080, -182.9254608, 180.1837158, -364.3082886, 364.2873535

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1718714, upper bound: 318.1727022
time: 9.38 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1783165, upper bound: 318.1787167
time: 10.14 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -201.7486572, 159.7551727, -202.8216553, 160.6661682, -362.4148254, 362.5768433
1: -170.0928955, 141.6844482, -170.9626465, 142.4190216, -312.5119019, 312.6470947
2: -222.1750488, 143.9926147, -223.4220886, 144.6844025, -366.8593750, 367.4147034
3: -235.9363251, 124.5760193, -237.1017303, 125.1580048, -361.0942688, 361.6777344
4: -216.7253571, 165.7321167, -217.8985748, 166.5117493, -383.2370911, 383.6306763
5: -193.5281372, 150.6325836, -194.5508575, 151.2988739, -344.8270264, 345.1834412
6: -185.3204346, 178.8934479, -186.3264008, 179.8486481, -365.1690369, 365.2198181
7: -202.6959381, 169.7283020, -203.7136383, 170.5898743, -373.2858276, 373.4419556
8: -243.8268127, 166.6226654, -245.1721802, 167.5205536, -411.3473206, 411.7948303
9: -184.1246033, 181.3619080, -185.0407104, 182.3162537, -366.4407959, 366.4026184

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1718714, upper bound: 318.1727022
time: 10.60 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1783165, upper bound: 318.1787166
time: 9.75 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -204.1518860, 161.7150726, -200.3996582, 158.6906281, -362.8424683, 362.1147461
1: -172.0771484, 143.3549652, -168.9624329, 140.7354889, -312.8126221, 312.3173828
2: -224.8767548, 145.6356659, -220.6999054, 143.0297089, -367.9064636, 366.3355713
3: -238.6658020, 125.9837494, -234.3511200, 123.7395706, -362.4053650, 360.3348083
4: -219.3159637, 167.6087952, -215.2885590, 164.6194305, -383.9353943, 382.8973389
5: -195.8265228, 152.3001862, -192.2327881, 149.6169891, -345.4435120, 344.5329590
6: -187.5478821, 181.0230103, -184.0817566, 177.7034760, -365.2512817, 365.1047668
7: -205.0476685, 171.7090454, -201.3444214, 168.5936127, -373.6412964, 373.0534363
8: -246.7667084, 168.6043701, -242.2112885, 165.5245972, -412.2913208, 410.8156738
9: -186.2518005, 183.5061493, -182.8972168, 180.1564331, -366.4082336, 366.4033203

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 53

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1470386, upper bound: 318.1510624
time: 9.34 seconds

## Relational analysis of IS_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1553006, upper bound: 318.1616771
time: 9.66 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -204.1518860, 161.7150726, -202.8216553, 160.6661682, -364.8180542, 364.5367432
1: -172.0771484, 143.3549652, -170.9626465, 142.4190216, -314.4961548, 314.3176270
2: -224.8767548, 145.6356659, -223.4220886, 144.6844025, -369.5611267, 369.0577393
3: -238.6658020, 125.9837494, -237.1017303, 125.1580048, -363.8237915, 363.0854492
4: -219.3159637, 167.6087952, -217.8985748, 166.5117493, -385.8276978, 385.5073853
5: -195.8265228, 152.3001862, -194.5508575, 151.2988739, -347.1253967, 346.8510437
6: -187.5478821, 181.0230103, -186.3264008, 179.8486481, -367.3964844, 367.3493347
7: -205.0476685, 171.7090454, -203.7136383, 170.5898743, -375.6375427, 375.4226685
8: -246.7667084, 168.6043701, -245.1721802, 167.5205536, -414.2872620, 413.7764893
9: -186.2518005, 183.5061493, -185.0407104, 182.3162537, -368.5680542, 368.5468445

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 53

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 53

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1470386, upper bound: 318.1510624
time: 10.52 seconds

## Relational analysis of IS_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1553006, upper bound: 318.1616771
time: 9.27 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -199.2002716, 157.7405243, -195.1947021, 154.6142731, -353.8145447, 352.9352417
1: -167.9598083, 139.8940277, -164.6117554, 137.0828552, -305.0426636, 304.5057983
2: -219.3876801, 142.1737061, -215.1534271, 139.2315979, -358.6192627, 357.3271484
3: -232.9416351, 122.9999542, -228.3048706, 120.4562912, -353.3979187, 351.3048096
4: -214.0026855, 163.6305237, -209.9033356, 160.2079163, -374.2106018, 373.5338440
5: -191.0789185, 148.7147217, -187.2476654, 145.5124359, -336.5913391, 335.9623718
6: -182.9782410, 176.6448517, -179.4261932, 173.2020874, -356.1803284, 356.0710144
7: -200.1430664, 167.5855560, -196.1527557, 164.2102356, -364.3532715, 363.7383118
8: -240.7716980, 164.5447388, -236.1186829, 161.2922974, -402.0639648, 400.6634216
9: -181.8015289, 179.0815887, -178.1711121, 175.5274200, -357.3289490, 357.2526855

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1437084, upper bound: 318.1409097
time: 9.95 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1552794, upper bound: 318.1510600
time: 8.90 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -199.2002716, 157.7405243, -198.1507416, 157.0154266, -356.2156982, 355.8912659
1: -167.9598083, 139.8940277, -167.0614624, 139.1473846, -307.1071777, 306.9554443
2: -219.3876801, 142.1737061, -218.4706421, 141.2795715, -360.6672363, 360.6443481
3: -232.9416351, 122.9999542, -231.6992188, 122.2084656, -355.1500854, 354.6991577
4: -214.0026855, 163.6305237, -213.1110535, 162.5469971, -376.5496826, 376.7415771
5: -191.0789185, 148.7147217, -190.0676727, 147.5897369, -338.6686401, 338.7823486
6: -182.9782410, 176.6448517, -182.1749115, 175.8325043, -358.8107300, 358.8197632
7: -200.1430664, 167.5855560, -199.0747681, 166.6582794, -366.8013306, 366.6603394
8: -240.7716980, 164.5447388, -239.7484131, 163.7449341, -404.5166321, 404.2931519
9: -181.8015289, 179.0815887, -180.8128052, 178.1820221, -359.9835510, 359.8943787

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 212

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1437084, upper bound: 318.1409097
time: 9.40 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1552794, upper bound: 318.1510600
time: 8.36 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -201.7379150, 159.8072052, -195.1930084, 154.6128540, -356.3507690, 355.0002136
1: -170.0571136, 141.6597900, -164.6102295, 137.0816803, -307.1387634, 306.2700195
2: -222.2375641, 143.9138794, -215.1515503, 139.2304077, -361.4679565, 359.0654297
3: -235.8313141, 124.4910660, -228.3029480, 120.4551620, -356.2864685, 352.7940063
4: -216.7390289, 165.6190948, -209.9015198, 160.2064514, -376.9454956, 375.5205994
5: -193.5068970, 150.4838409, -187.2460175, 145.5110931, -339.0179749, 337.7298584
6: -185.3315887, 178.8947906, -179.4246979, 173.2005463, -358.5321350, 358.3194885
7: -202.6311646, 169.6802368, -196.1510620, 164.2088013, -366.8399658, 365.8312683
8: -243.8740234, 166.6385040, -236.1165466, 161.2907104, -405.1647339, 402.7550659
9: -184.0529327, 181.3471680, -178.1695862, 175.5258484, -359.5787354, 359.5167542

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1348097, upper bound: 318.1353321
time: 7.31 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1384266, upper bound: 318.1384266
time: 9.11 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -201.7379150, 159.8072052, -198.1507416, 157.0154266, -358.7533569, 357.9579468
1: -170.0571136, 141.6597900, -167.0614624, 139.1473846, -309.2044373, 308.7212219
2: -222.2375641, 143.9138794, -218.4706421, 141.2795715, -363.5171204, 362.3845215
3: -235.8313141, 124.4910660, -231.6992188, 122.2084656, -358.0397339, 356.1902771
4: -216.7390289, 165.6190948, -213.1110535, 162.5469971, -379.2860107, 378.7301636
5: -193.5068970, 150.4838409, -190.0676727, 147.5897369, -341.0965881, 340.5514832
6: -185.3315887, 178.8947906, -182.1749115, 175.8325043, -361.1640930, 361.0697021
7: -202.6311646, 169.6802368, -199.0747681, 166.6582794, -369.2894287, 368.7549438
8: -243.8740234, 166.6385040, -239.7484131, 163.7449341, -407.6189575, 406.3869019
9: -184.0529327, 181.3471680, -180.8128052, 178.1820221, -362.2349243, 362.1599731

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 136

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1348097, upper bound: 318.1353321
time: 7.28 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1384266, upper bound: 318.1384266
time: 7.37 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 44.07 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 44.07
Output dim: 1, lower bound: -318.1718714, upper bound: 318.1727022
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 44.07
Output dim: 1, lower bound: -318.1783165, upper bound: 318.1787167
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 44.07
Output dim: 1, lower bound: -318.1718714, upper bound: 318.1727022
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 44.07
Output dim: 1, lower bound: -318.1783165, upper bound: 318.1787166
IS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 44.07
Output dim: 1, lower bound: -318.1470386, upper bound: 318.1510624
IS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 44.07
Output dim: 1, lower bound: -318.1553006, upper bound: 318.1616771
IS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 44.07
Output dim: 1, lower bound: -318.1470386, upper bound: 318.1510624
IS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 44.07
Output dim: 1, lower bound: -318.1553006, upper bound: 318.1616771
IS_B2_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 44.07
Output dim: 1, lower bound: -318.1437084, upper bound: 318.1409097
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 44.07
Output dim: 1, lower bound: -318.1552794, upper bound: 318.1510600
IS_B2_A1_B2_A1, status: Status.VERIFIED, split count: 4, time: 44.07
Output dim: 1, lower bound: -318.1437084, upper bound: 318.1409097
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 44.07
Output dim: 1, lower bound: -318.1552794, upper bound: 318.1510600
IS_B2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 44.07
Output dim: 1, lower bound: -318.1348097, upper bound: 318.1353321
IS_B2_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 44.07
Output dim: 1, lower bound: -318.1384266, upper bound: 318.1384266
IS_B2_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 44.07
Output dim: 1, lower bound: -318.1348097, upper bound: 318.1353321
IS_B2_A2_B2_A2, status: Status.VERIFIED, split count: 4, time: 44.07
Output dim: 1, lower bound: -318.1384266, upper bound: 318.1384266

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -196.4129181, 155.5406342, -197.6102600, 156.4873352, -352.9001770, 353.1508484
1: -165.6728516, 137.9382324, -166.6398926, 138.7765045, -304.4493408, 304.5781250
2: -216.2559509, 140.2201538, -217.6155243, 141.0498352, -357.3057861, 357.8356628
3: -229.7447205, 121.3060303, -231.1010437, 122.0251389, -351.7698364, 352.4070129
4: -211.0579376, 161.3769684, -212.3077240, 162.3360748, -373.3939514, 373.6846619
5: -188.4514465, 146.6773376, -189.5694275, 147.5491486, -336.0006104, 336.2467651
6: -180.4556885, 174.1737061, -181.5251617, 175.2316895, -355.6873779, 355.6988525
7: -197.3137360, 165.2317352, -198.5356598, 166.2440338, -363.5577698, 363.7673950
8: -237.4391785, 162.3589020, -238.8569336, 163.2692413, -400.7084351, 401.2158203
9: -179.2950897, 176.6215973, -180.3587494, 177.6621094, -356.9572144, 356.9802551

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 212

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1892958, upper bound: 318.1919896
time: 10.34 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1892958, upper bound: 318.1924691
time: 11.18 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -198.7752075, 157.4042206, -199.7436829, 158.1725311, -356.9476929, 357.1478882
1: -167.6228180, 139.5940857, -168.4180450, 140.2741394, -307.8969116, 308.0120544
2: -218.8805084, 141.8834534, -219.9726715, 142.5636444, -361.4441528, 361.8560791
3: -232.4825134, 122.7492676, -233.5889893, 123.3366852, -355.8191833, 356.3382568
4: -213.5547791, 163.2969666, -214.5886688, 164.0822144, -377.6369629, 377.8856201
5: -190.6893616, 148.4258881, -191.6082611, 149.1311798, -339.8204956, 340.0341492
6: -182.5982361, 176.2605286, -183.4811096, 177.1218567, -359.7200928, 359.7416382
7: -199.6986542, 167.2170410, -200.6824951, 168.0394440, -367.7380981, 367.8995361
8: -240.2585602, 164.2339630, -241.4232178, 164.9972839, -405.2558289, 405.6571655
9: -181.4189453, 178.7067719, -182.2999878, 179.5699768, -360.9888916, 361.0067444

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1908527, upper bound: 318.1930799
time: 10.25 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1908527, upper bound: 318.1950037
time: 9.37 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 20.90 seconds
IS_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 20.90
Output dim: 1, lower bound: -318.1892958, upper bound: 318.1919896
IS_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 20.90
Output dim: 1, lower bound: -318.1892958, upper bound: 318.1924691
IS_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 20.90
Output dim: 1, lower bound: -318.1908527, upper bound: 318.1930799
IS_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 20.90
Output dim: 1, lower bound: -318.1908527, upper bound: 318.1950037
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 20.90
Output dim: 1, lower bound: -318.1718714, upper bound: 318.1727022
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 20.90
Output dim: 1, lower bound: -318.1783165, upper bound: 318.1787166
IS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 20.90
Output dim: 1, lower bound: -318.1470386, upper bound: 318.1510624
IS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 20.90
Output dim: 1, lower bound: -318.1553006, upper bound: 318.1616771
IS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 20.90
Output dim: 1, lower bound: -318.1470386, upper bound: 318.1510624
IS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 20.90
Output dim: 1, lower bound: -318.1553006, upper bound: 318.1616771
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 20.90
Output dim: 1, lower bound: -318.1552794, upper bound: 318.1510600
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 20.90
Output dim: 1, lower bound: -318.1552794, upper bound: 318.1510600
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=319.7423400878906
rel_dist={1: [-318.23529533371016, 318.23529533356407]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1940111, upper bound: 318.1946602
time: 12.00 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1921905, upper bound: 318.1921906
time: 7.03 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.16 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 19.16
Output dim: 1, lower bound: -318.1940111, upper bound: 318.1946602
IS_B2, status: Status.UNKNOWN, split count: 1, time: 19.16
Output dim: 1, lower bound: -318.1921905, upper bound: 318.1921906

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -206.9136353, 163.8058472, -205.5787811, 162.7531281, -369.6666870, 369.3846436
1: -174.4282379, 145.3141937, -173.3099365, 144.3747864, -318.8029480, 318.6240234
2: -227.8242950, 147.6721191, -226.3644714, 146.7181244, -374.5424194, 374.0365906
3: -242.0703125, 127.7758865, -240.5008850, 126.9480591, -369.0183716, 368.2767639
4: -222.2664337, 169.9702454, -220.8441620, 168.8690796, -391.1354980, 390.8143921
5: -198.4908142, 154.4979553, -197.2106323, 153.4936523, -351.9844666, 351.7085876
6: -190.0688934, 183.4619141, -188.8430939, 182.2836456, -372.3525391, 372.3049927
7: -207.8973694, 174.0781555, -206.5591736, 172.9551849, -380.8525391, 380.6373291
8: -250.0145874, 170.8148804, -248.4152527, 169.7273712, -419.7419434, 419.2301331
9: -188.8386993, 185.9809875, -187.6236420, 184.7870789, -373.6257324, 373.6046143

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1921905, upper bound: 318.1921906
time: 10.31 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1921905, upper bound: 318.1921906
time: 8.51 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -200.0794983, 158.4038239, -200.4977570, 158.7754822, -358.8549500, 358.9015503
1: -168.7073822, 140.5136566, -169.0623627, 140.8095551, -309.5169373, 309.5760193
2: -220.3510590, 142.7966919, -220.9532623, 143.0125580, -363.3635864, 363.7499390
3: -234.0423279, 123.5514832, -234.6011963, 123.7453156, -357.7876587, 358.1525879
4: -214.9655457, 164.3338928, -215.5938873, 164.5584106, -379.5239563, 379.9277344
5: -191.9248047, 149.3576813, -192.3430634, 149.4841919, -341.4089966, 341.7007446
6: -183.7890930, 177.4348602, -184.3029785, 177.8934937, -361.6825867, 361.7378540
7: -201.0521240, 168.3347321, -201.4940186, 168.6785126, -369.7306519, 369.8287048
8: -241.8211823, 165.2419128, -242.4742432, 165.5976562, -407.4188232, 407.7161255
9: -182.6093750, 179.8667908, -183.0123749, 180.2741089, -362.8834534, 362.8790894

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1921905, upper bound: 318.1921905
time: 7.70 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1921905, upper bound: 318.1921906
time: 8.51 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 17.49 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 17.49
Output dim: 1, lower bound: -318.1921905, upper bound: 318.1921906
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 17.49
Output dim: 1, lower bound: -318.1921905, upper bound: 318.1921906
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 17.49
Output dim: 1, lower bound: -318.1921905, upper bound: 318.1921905
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 17.49
Output dim: 1, lower bound: -318.1921905, upper bound: 318.1921906

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -205.5787811, 162.7531281, -205.5787811, 162.7531281, -368.3319092, 368.3319092
1: -173.3099365, 144.3747864, -173.3099365, 144.3747864, -317.6846313, 317.6846313
2: -226.3644714, 146.7181244, -226.3644714, 146.7181244, -373.0825806, 373.0825806
3: -240.5008850, 126.9480591, -240.5008850, 126.9480591, -367.4489441, 367.4489441
4: -220.8441620, 168.8690796, -220.8441620, 168.8690796, -389.7132568, 389.7132568
5: -197.2106323, 153.4936523, -197.2106323, 153.4936523, -350.7042847, 350.7042847
6: -188.8430939, 182.2836456, -188.8430939, 182.2836456, -371.1267395, 371.1267395
7: -206.5591736, 172.9551849, -206.5591736, 172.9551849, -379.5143433, 379.5143433
8: -248.4152527, 169.7273712, -248.4152527, 169.7273712, -418.1426392, 418.1426392
9: -187.6236420, 184.7870789, -187.6236420, 184.7870789, -372.4107056, 372.4107056

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1602728, upper bound: 318.1598970
time: 11.44 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1420212, upper bound: 318.1445997
time: 11.20 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -200.4977570, 158.7754822, -205.5787811, 162.7531281, -363.2508850, 364.3542480
1: -169.0623627, 140.8095551, -173.3099365, 144.3747864, -313.4371033, 314.1194763
2: -220.9532623, 143.0125580, -226.3644714, 146.7181244, -367.6713867, 369.3770142
3: -234.6011963, 123.7453156, -240.5008850, 126.9480591, -361.5492249, 364.2461853
4: -215.5938873, 164.5584106, -220.8441620, 168.8690796, -384.4629517, 385.4025879
5: -192.3430634, 149.4841919, -197.2106323, 153.4936523, -345.8367310, 346.6948242
6: -184.3029785, 177.8934937, -188.8430939, 182.2836456, -366.5866089, 366.7365723
7: -201.4940186, 168.6785126, -206.5591736, 172.9551849, -374.4492188, 375.2376709
8: -242.4742432, 165.5976562, -248.4152527, 169.7273712, -412.2015991, 414.0129089
9: -183.0123749, 180.2741089, -187.6236420, 184.7870789, -367.7994080, 367.8977356

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1496926, upper bound: 318.1522835
time: 11.08 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1491686, upper bound: 318.1513814
time: 11.29 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -205.5419159, 162.7239227, -200.4977570, 158.7754822, -364.3173828, 363.2216797
1: -173.2802124, 144.3486023, -169.0623627, 140.8095551, -314.0897827, 313.4109192
2: -226.3239594, 146.6921387, -220.9532623, 143.0125580, -369.3365173, 367.6453857
3: -240.4571228, 126.9260178, -234.6011963, 123.7453156, -364.2024536, 361.5271912
4: -220.8048096, 168.8381805, -215.5938873, 164.5584106, -385.3632202, 384.4320374
5: -197.1750336, 153.4668579, -192.3430634, 149.4841919, -346.6592407, 345.8099060
6: -188.8091888, 182.2513580, -184.3029785, 177.8934937, -366.7026978, 366.5543213
7: -206.5233459, 172.9234467, -201.4940186, 168.6785126, -375.2018433, 374.4174805
8: -248.3720703, 169.6970825, -242.4742432, 165.5976562, -413.9697266, 412.1713257
9: -187.5900879, 184.7548065, -183.0123749, 180.2741089, -367.8641663, 367.7671204

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1492541, upper bound: 318.1477626
time: 9.90 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1460623, upper bound: 318.1460623
time: 7.50 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -200.4977570, 158.7754822, -200.4977570, 158.7754822, -359.2732239, 359.2732239
1: -169.0623627, 140.8095551, -169.0623627, 140.8095551, -309.8719177, 309.8719177
2: -220.9532623, 143.0125580, -220.9532623, 143.0125580, -363.9657898, 363.9657898
3: -234.6011963, 123.7453156, -234.6011963, 123.7453156, -358.3464661, 358.3464661
4: -215.5938873, 164.5584106, -215.5938873, 164.5584106, -380.1522827, 380.1522827
5: -192.3430634, 149.4841919, -192.3430634, 149.4841919, -341.8272705, 341.8272705
6: -184.3029785, 177.8934937, -184.3029785, 177.8934937, -362.1964722, 362.1964722
7: -201.4940186, 168.6785126, -201.4940186, 168.6785126, -370.1725159, 370.1725159
8: -242.4742432, 165.5976562, -242.4742432, 165.5976562, -408.0718994, 408.0718994
9: -183.0123749, 180.2741089, -183.0123749, 180.2741089, -363.2864380, 363.2864380

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1403749, upper bound: 318.1420369
time: 8.14 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1383032, upper bound: 318.1383031
time: 7.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 16.95 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 16.95
Output dim: 1, lower bound: -318.1602728, upper bound: 318.1598970
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 16.95
Output dim: 1, lower bound: -318.1420212, upper bound: 318.1445997
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 16.95
Output dim: 1, lower bound: -318.1496926, upper bound: 318.1522835
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 16.95
Output dim: 1, lower bound: -318.1491686, upper bound: 318.1513814
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 16.95
Output dim: 1, lower bound: -318.1492541, upper bound: 318.1477626
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 16.95
Output dim: 1, lower bound: -318.1460623, upper bound: 318.1460623
IS_B2_A2_B1, status: Status.VERIFIED, split count: 3, time: 16.95
Output dim: 1, lower bound: -318.1403749, upper bound: 318.1420369
IS_B2_A2_B2, status: Status.VERIFIED, split count: 3, time: 16.95
Output dim: 1, lower bound: -318.1383032, upper bound: 318.1383031

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -200.2054291, 158.5080261, -201.7482605, 159.7271118, -359.9325562, 360.2561951
1: -168.8321838, 140.5958710, -170.1179199, 141.6809540, -310.5131226, 310.7138062
2: -220.4826660, 142.9130096, -222.1716309, 144.0058594, -364.4885254, 365.0845642
3: -234.1737061, 123.6467285, -235.9902649, 124.5946503, -358.7682495, 359.6369629
4: -215.1185913, 164.5028687, -216.7627716, 165.7566528, -380.8752136, 381.2655334
5: -192.0519257, 149.4883881, -193.5331879, 150.6384735, -342.6903687, 343.0214844
6: -183.9443359, 177.5594482, -185.3509827, 178.9159393, -362.8602905, 362.9104309
7: -201.1715393, 168.4461517, -202.7186432, 169.7409668, -370.9123840, 371.1647949
8: -242.0081329, 165.4156342, -243.8479919, 166.6540070, -408.6620789, 409.2635803
9: -182.7558594, 180.0000763, -184.1536560, 181.3745422, -364.1304016, 364.1537476

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2271515, upper bound: 318.2271515
time: 11.04 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2271515, upper bound: 318.2271515
time: 10.27 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -201.3956451, 159.4595795, -199.4304352, 157.8894958, -359.2851257, 358.8900146
1: -169.8973541, 141.4384460, -168.1864929, 140.0393677, -309.9366455, 309.6249084
2: -221.8314056, 143.7516479, -219.6385193, 142.3606110, -364.1919861, 363.3901367
3: -235.4926147, 124.3828659, -233.2313232, 123.1576614, -358.6502686, 357.6141968
4: -216.4551392, 165.4802399, -214.2925568, 163.8632812, -380.3183899, 379.7727966
5: -193.1856232, 150.3539124, -191.3069916, 148.9012604, -342.0868835, 341.6608887
6: -185.0551910, 178.6555481, -183.2352600, 176.8716888, -361.9268799, 361.8907776
7: -202.3376617, 169.4338684, -200.3824615, 167.7923889, -370.1300659, 369.8162842
8: -243.5343323, 166.4673615, -241.0905457, 164.7882690, -408.3226013, 407.5578308
9: -183.8461304, 181.1177826, -182.0484619, 179.3120117, -363.1581421, 363.1662598

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2271515, upper bound: 318.2271515
time: 11.84 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.2271515, upper bound: 318.2271514
time: 11.18 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -197.0920563, 156.1028595, -200.4309692, 158.7160034, -355.8080444, 356.5338135
1: -166.2039642, 138.4163513, -168.9890137, 140.7574310, -306.9613647, 307.4053650
2: -217.2284393, 140.5843964, -220.7341919, 143.0513000, -360.2796936, 361.3186035
3: -230.5566864, 121.6331635, -234.3875580, 123.7591095, -354.3157959, 356.0207214
4: -211.9394379, 161.7639008, -215.3217163, 164.6451569, -376.5845947, 377.0856018
5: -189.0701904, 146.9336395, -192.2644653, 149.6413422, -338.7115173, 339.1981201
6: -181.1704254, 174.8804779, -184.1105347, 177.7305756, -358.9009399, 358.9910278
7: -198.0636749, 165.8089752, -201.3753967, 168.6199493, -366.6836243, 367.1843872
8: -238.3922729, 162.8326416, -242.2480469, 165.5495605, -403.9418335, 405.0806885
9: -179.9032135, 177.2254486, -182.9254608, 180.1837158, -360.0869141, 360.1509094

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1491686, upper bound: 318.1513813
time: 11.94 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1491686, upper bound: 318.1513814
time: 9.69 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -191.6178741, 151.8026428, -202.8216553, 160.6661682, -352.2840576, 354.6242981
1: -161.6115723, 134.5688019, -170.9626465, 142.4190216, -304.0305786, 305.5314331
2: -211.2456818, 136.6709595, -223.4220886, 144.6844025, -355.9300232, 360.0930481
3: -224.0701599, 118.2450714, -237.1017303, 125.1580048, -349.2281494, 355.3468018
4: -206.0498047, 157.2676086, -217.8985748, 166.5117493, -372.5615234, 375.1661987
5: -183.8065948, 142.8195343, -194.5508575, 151.2988739, -335.1054688, 337.3703918
6: -176.1335754, 170.0393982, -186.3264008, 179.8486481, -355.9822388, 356.3657532
7: -192.5462952, 161.1937408, -203.7136383, 170.5898743, -363.1361389, 364.9073792
8: -231.8375397, 158.3849640, -245.1721802, 167.5205536, -399.3580627, 403.5570984
9: -174.8881683, 172.3228607, -185.0407104, 182.3162537, -357.2044067, 357.3635864

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1491686, upper bound: 318.1513813
time: 10.94 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1491686, upper bound: 318.1513814
time: 9.91 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -200.4212494, 158.7082825, -197.0920563, 156.1028595, -356.5240784, 355.8003235
1: -168.9811401, 140.7504883, -166.2039642, 138.4163513, -307.3974915, 306.9544373
2: -220.7235260, 143.0444489, -217.2284393, 140.5843964, -361.3079224, 360.2727966
3: -234.3760223, 123.7532883, -230.5566864, 121.6331635, -356.0091858, 354.3099670
4: -215.3113251, 164.6369476, -211.9394379, 161.7639008, -377.0752258, 376.5763855
5: -192.2550507, 149.6342621, -189.0701904, 146.9336395, -339.1886902, 338.7044678
6: -184.1015625, 177.7220459, -181.1704254, 174.8804779, -358.9820557, 358.8924255
7: -201.3659363, 168.6115417, -198.0636749, 165.8089752, -367.1749268, 366.6752319
8: -242.2366333, 165.5415497, -238.3922729, 162.8326416, -405.0692749, 403.9338379
9: -182.9166412, 180.1751709, -179.9032135, 177.2254486, -360.1420898, 360.0783691

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1513813, upper bound: 318.1491687
time: 12.36 seconds

## Relational analysis of IS_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1513813, upper bound: 318.1491687
time: 10.58 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -202.8216553, 160.6661682, -191.6178741, 151.8026428, -354.6242981, 352.2840576
1: -170.9626465, 142.4190216, -161.6115723, 134.5688019, -305.5314331, 304.0305786
2: -223.4220886, 144.6844025, -211.2456818, 136.6709595, -360.0930481, 355.9300232
3: -237.1017303, 125.1580048, -224.0701599, 118.2450714, -355.3468018, 349.2281494
4: -217.8985748, 166.5117493, -206.0498047, 157.2676086, -375.1661987, 372.5615234
5: -194.5508575, 151.2988739, -183.8065948, 142.8195343, -337.3703918, 335.1054688
6: -186.3264008, 179.8486481, -176.1335754, 170.0393982, -356.3657532, 355.9822388
7: -203.7136383, 170.5898743, -192.5462952, 161.1937408, -364.9073792, 363.1361389
8: -245.1721802, 167.5205536, -231.8375397, 158.3849640, -403.5570984, 399.3580627
9: -185.0407104, 182.3162537, -174.8881683, 172.3228607, -357.3635864, 357.2044067

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1513813, upper bound: 318.1491687
time: 10.25 seconds

## Relational analysis of IS_B2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1513813, upper bound: 318.1491686
time: 9.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.13 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 1, lower bound: -318.2271515, upper bound: 318.2271515
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 1, lower bound: -318.2271515, upper bound: 318.2271515
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 1, lower bound: -318.2271515, upper bound: 318.2271515
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 1, lower bound: -318.2271515, upper bound: 318.2271514
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 1, lower bound: -318.1491686, upper bound: 318.1513813
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 1, lower bound: -318.1491686, upper bound: 318.1513814
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 1, lower bound: -318.1491686, upper bound: 318.1513813
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 1, lower bound: -318.1491686, upper bound: 318.1513814
IS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 1, lower bound: -318.1513813, upper bound: 318.1491687
IS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 1, lower bound: -318.1513813, upper bound: 318.1491687
IS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 1, lower bound: -318.1513813, upper bound: 318.1491687
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.13
Output dim: 1, lower bound: -318.1513813, upper bound: 318.1491686

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -200.2054291, 158.5080261, -200.2054291, 158.5080261, -358.7134399, 358.7134399
1: -168.8321838, 140.5958710, -168.8321838, 140.5958710, -309.4280396, 309.4280396
2: -220.4826660, 142.9130096, -220.4826660, 142.9130096, -363.3956299, 363.3956299
3: -234.1737061, 123.6467285, -234.1737061, 123.6467285, -357.8203735, 357.8203735
4: -215.1185913, 164.5028687, -215.1185913, 164.5028687, -379.6213989, 379.6213989
5: -192.0519257, 149.4883881, -192.0519257, 149.4883881, -341.5402527, 341.5402527
6: -183.9443359, 177.5594482, -183.9443359, 177.5594482, -361.5037842, 361.5037842
7: -201.1715393, 168.4461517, -201.1715393, 168.4461517, -369.6176453, 369.6176453
8: -242.0081329, 165.4156342, -242.0081329, 165.4156342, -407.4236755, 407.4236755
9: -182.7558594, 180.0000763, -182.7558594, 180.0000763, -362.7559204, 362.7559204

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_B1_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1122163, upper bound: 318.1088891
time: 10.77 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1039876, upper bound: 318.1036463
time: 8.69 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -200.2054291, 158.5080261, -201.3956451, 159.4595795, -359.6650085, 359.9036865
1: -168.8321838, 140.5958710, -169.8973541, 141.4384460, -310.2705688, 310.4932251
2: -220.4826660, 142.9130096, -221.8314056, 143.7516479, -364.2343140, 364.7443237
3: -234.1737061, 123.6467285, -235.4926147, 124.3828659, -358.5565186, 359.1393127
4: -215.1185913, 164.5028687, -216.4551392, 165.4802399, -380.5988159, 380.9579468
5: -192.0519257, 149.4883881, -193.1856232, 150.3539124, -342.4058228, 342.6739502
6: -183.9443359, 177.5594482, -185.0551910, 178.6555481, -362.5998840, 362.6146240
7: -201.1715393, 168.4461517, -202.3376617, 169.4338684, -370.6053162, 370.7838135
8: -242.0081329, 165.4156342, -243.5343323, 166.4673615, -408.4754333, 408.9499512
9: -182.7558594, 180.0000763, -183.8461304, 181.1177826, -363.8736572, 363.8461914

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of IS_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1119891, upper bound: 318.1141671
time: 10.21 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1039876, upper bound: 318.1036463
time: 7.77 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -201.3956451, 159.4595795, -200.2054291, 158.5080261, -359.9036865, 359.6650085
1: -169.8973541, 141.4384460, -168.8321838, 140.5958710, -310.4932251, 310.2705688
2: -221.8314056, 143.7516479, -220.4826660, 142.9130096, -364.7443237, 364.2343140
3: -235.4926147, 124.3828659, -234.1737061, 123.6467285, -359.1393127, 358.5565186
4: -216.4551392, 165.4802399, -215.1185913, 164.5028687, -380.9579468, 380.5988159
5: -193.1856232, 150.3539124, -192.0519257, 149.4883881, -342.6739502, 342.4058228
6: -185.0551910, 178.6555481, -183.9443359, 177.5594482, -362.6146240, 362.5998840
7: -202.3376617, 169.4338684, -201.1715393, 168.4461517, -370.7838135, 370.6053162
8: -243.5343323, 166.4673615, -242.0081329, 165.4156342, -408.9499512, 408.4754333
9: -183.8461304, 181.1177826, -182.7558594, 180.0000763, -363.8461914, 363.8736572

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1117997, upper bound: 318.1086405
time: 11.02 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1032774, upper bound: 318.1032774
time: 7.36 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -201.3956451, 159.4595795, -201.3956451, 159.4595795, -360.8552246, 360.8552246
1: -169.8973541, 141.4384460, -169.8973541, 141.4384460, -311.3357239, 311.3357239
2: -221.8314056, 143.7516479, -221.8314056, 143.7516479, -365.5829773, 365.5829773
3: -235.4926147, 124.3828659, -235.4926147, 124.3828659, -359.8754578, 359.8754578
4: -216.4551392, 165.4802399, -216.4551392, 165.4802399, -381.9353638, 381.9353638
5: -193.1856232, 150.3539124, -193.1856232, 150.3539124, -343.5395508, 343.5395508
6: -185.0551910, 178.6555481, -185.0551910, 178.6555481, -363.7107544, 363.7107544
7: -202.3376617, 169.4338684, -202.3376617, 169.4338684, -371.7715454, 371.7715454
8: -243.5343323, 166.4673615, -243.5343323, 166.4673615, -410.0016785, 410.0016785
9: -183.8461304, 181.1177826, -183.8461304, 181.1177826, -364.9639282, 364.9639282

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1117997, upper bound: 318.1086405
time: 12.68 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1032774, upper bound: 318.1032774
time: 9.18 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -195.1947021, 154.6142731, -200.4309692, 158.7160034, -353.9107056, 355.0452271
1: -164.6117554, 137.0828552, -168.9890137, 140.7574310, -305.3692017, 306.0718689
2: -215.1534271, 139.2315979, -220.7341919, 143.0513000, -358.2047119, 359.9657898
3: -228.3048706, 120.4562912, -234.3875580, 123.7591095, -352.0639648, 354.8438416
4: -209.9033356, 160.2079163, -215.3217163, 164.6451569, -374.5484619, 375.5296021
5: -187.2476654, 145.5124359, -192.2644653, 149.6413422, -336.8889465, 337.7769165
6: -179.4261932, 173.2020874, -184.1105347, 177.7305756, -357.1567078, 357.3126221
7: -196.1527557, 164.2102356, -201.3753967, 168.6199493, -364.7727051, 365.5856323
8: -236.1186829, 161.2922974, -242.2480469, 165.5495605, -401.6682434, 403.5403442
9: -178.1711121, 175.5274200, -182.9254608, 180.1837158, -358.3548279, 358.4528809

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1411258, upper bound: 318.1432549
time: 11.14 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1418722, upper bound: 318.1444558
time: 12.68 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -198.1507416, 157.0154266, -200.4309692, 158.7160034, -356.8666992, 357.4463806
1: -167.0614624, 139.1473846, -168.9890137, 140.7574310, -307.8188782, 308.1363831
2: -218.4706421, 141.2795715, -220.7341919, 143.0513000, -361.5218811, 362.0137634
3: -231.6992188, 122.2084656, -234.3875580, 123.7591095, -355.4583130, 356.5960083
4: -213.1110535, 162.5469971, -215.3217163, 164.6451569, -377.7562256, 377.8687134
5: -190.0676727, 147.5897369, -192.2644653, 149.6413422, -339.7089233, 339.8541870
6: -182.1749115, 175.8325043, -184.1105347, 177.7305756, -359.9054871, 359.9430542
7: -199.0747681, 166.6582794, -201.3753967, 168.6199493, -367.6947021, 368.0336914
8: -239.7484131, 163.7449341, -242.2480469, 165.5495605, -405.2979736, 405.9929810
9: -180.8128052, 178.1820221, -182.9254608, 180.1837158, -360.9965210, 361.1074829

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 114

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1494045, upper bound: 318.1519369
time: 11.42 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1496873, upper bound: 318.1522648
time: 11.59 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -195.1518860, 154.5806885, -202.8216553, 160.6661682, -355.8180542, 357.4023438
1: -164.5738983, 137.0524292, -170.9626465, 142.4190216, -306.9929199, 308.0150757
2: -215.1064148, 139.2028656, -223.4220886, 144.6844025, -359.7907410, 362.6249390
3: -228.2539673, 120.4299011, -237.1017303, 125.1580048, -353.4119873, 357.5316162
4: -209.8575134, 160.1728210, -217.8985748, 166.5117493, -376.3692627, 378.0714111
5: -187.2057343, 145.4814606, -194.5508575, 151.2988739, -338.5046082, 340.0323181
6: -179.3873444, 173.1640930, -186.3264008, 179.8486481, -359.2359619, 359.4904785
7: -196.1096191, 164.1752472, -203.7136383, 170.5898743, -366.6994629, 367.8888855
8: -236.0663910, 161.2580566, -245.1721802, 167.5205536, -403.5869446, 406.4302063
9: -178.1326447, 175.4897614, -185.0407104, 182.3162537, -360.4488525, 360.5304565

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1479790, upper bound: 318.1501185
time: 10.31 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1479684, upper bound: 318.1501146
time: 10.16 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -198.1507416, 157.0154266, -202.8216553, 160.6661682, -358.8168945, 359.8370972
1: -167.0614624, 139.1473846, -170.9626465, 142.4190216, -309.4804688, 310.1100464
2: -218.4706421, 141.2795715, -223.4220886, 144.6844025, -363.1549683, 364.7016296
3: -231.6992188, 122.2084656, -237.1017303, 125.1580048, -356.8572083, 359.3101807
4: -213.1110535, 162.5469971, -217.8985748, 166.5117493, -379.6228027, 380.4455566
5: -190.0676727, 147.5897369, -194.5508575, 151.2988739, -341.3665161, 342.1405640
6: -182.1749115, 175.8325043, -186.3264008, 179.8486481, -362.0235596, 362.1588745
7: -199.0747681, 166.6582794, -203.7136383, 170.5898743, -369.6646118, 370.3719177
8: -239.7484131, 163.7449341, -245.1721802, 167.5205536, -407.2688904, 408.9171143
9: -180.8128052, 178.1820221, -185.0407104, 182.3162537, -363.1290588, 363.2227173

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 157
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 118

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1487465, upper bound: 318.1508853
time: 11.04 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1491655, upper bound: 318.1513585
time: 11.51 seconds

## BFS IS instance: IS_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -200.4212494, 158.7082825, -195.1947021, 154.6142731, -355.0354919, 353.9029846
1: -168.9811401, 140.7504883, -164.6117554, 137.0828552, -306.0639954, 305.3622437
2: -220.7235260, 143.0444489, -215.1534271, 139.2315979, -359.9551392, 358.1978455
3: -234.3760223, 123.7532883, -228.3048706, 120.4562912, -354.8323059, 352.0581665
4: -215.3113251, 164.6369476, -209.9033356, 160.2079163, -375.5191956, 374.5402527
5: -192.2550507, 149.6342621, -187.2476654, 145.5124359, -337.7674866, 336.8819275
6: -184.1015625, 177.7220459, -179.4261932, 173.2020874, -357.3036499, 357.1481934
7: -201.3659363, 168.6115417, -196.1527557, 164.2102356, -365.5761719, 364.7642822
8: -242.2366333, 165.5415497, -236.1186829, 161.2922974, -403.5289307, 401.6602173
9: -182.9166412, 180.1751709, -178.1711121, 175.5274200, -358.4440613, 358.3462830

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_A1_B1_A1

### Relational analysis result of IS_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -318.1432549, upper bound: 318.1411258
time: 12.73 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2

### Relational analysis result of IS_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1444557, upper bound: 318.1418722
time: 11.90 seconds

## BFS IS instance: IS_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -200.4212494, 158.7082825, -198.1507416, 157.0154266, -357.4366455, 356.8589783
1: -168.9811401, 140.7504883, -167.0614624, 139.1473846, -308.1284790, 307.8119507
2: -220.7235260, 143.0444489, -218.4706421, 141.2795715, -362.0030823, 361.5150146
3: -234.3760223, 123.7532883, -231.6992188, 122.2084656, -356.5844727, 355.4524231
4: -215.3113251, 164.6369476, -213.1110535, 162.5469971, -377.8583374, 377.7479858
5: -192.2550507, 149.6342621, -190.0676727, 147.5897369, -339.8447876, 339.7019043
6: -184.1015625, 177.7220459, -182.1749115, 175.8325043, -359.9340820, 359.8969727
7: -201.3659363, 168.6115417, -199.0747681, 166.6582794, -368.0242310, 367.6863098
8: -242.2366333, 165.5415497, -239.7484131, 163.7449341, -405.9815674, 405.2899780
9: -182.9166412, 180.1751709, -180.8128052, 178.1820221, -361.0986633, 360.9879761

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 157
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 55
type: A, layer: 1, pos: 55
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 114

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_B2_A1_A1_B2_A1

### Relational analysis result of IS_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1519369, upper bound: 318.1494045
time: 11.02 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2

### Relational analysis result of IS_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -318.1522648, upper bound: 318.1496873
time: 12.03 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 41.25 seconds
IS_B1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1122163, upper bound: 318.1088891
IS_B1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1039876, upper bound: 318.1036463
IS_B1_A1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1119891, upper bound: 318.1141671
IS_B1_A1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1039876, upper bound: 318.1036463
IS_B1_A1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1117997, upper bound: 318.1086405
IS_B1_A1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1032774, upper bound: 318.1032774
IS_B1_A1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1117997, upper bound: 318.1086405
IS_B1_A1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1032774, upper bound: 318.1032774
IS_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1411258, upper bound: 318.1432549
IS_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1418722, upper bound: 318.1444558
IS_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1494045, upper bound: 318.1519369
IS_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1496873, upper bound: 318.1522648
IS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1479790, upper bound: 318.1501185
IS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1479684, upper bound: 318.1501146
IS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1487465, upper bound: 318.1508853
IS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1491655, upper bound: 318.1513585
IS_B2_A1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1432549, upper bound: 318.1411258
IS_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1444557, upper bound: 318.1418722
IS_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1519369, upper bound: 318.1494045
IS_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 41.25
Output dim: 1, lower bound: -318.1522648, upper bound: 318.1496873
IS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 41.25
Output dim: 1, lower bound: -318.1513813, upper bound: 318.1491687
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 41.25
Output dim: 1, lower bound: -318.1513813, upper bound: 318.1491686
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=319.7423400878906
rel_dist={1: [-318.2352719030525, 318.23527182273534]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1834.70 seconds
