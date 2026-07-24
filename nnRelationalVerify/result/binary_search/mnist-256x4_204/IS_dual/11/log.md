## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 460.407499041
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-256.7159119, 203.7015533, -256.7159119, 203.7015533, -460.4174805, 460.4174805)
1: (-215.5436707, 181.1157990, -215.5436707, 181.1157990, -396.6594543, 396.6594543)
2: (-282.8133240, 182.7726288, -282.8133240, 182.7726288, -465.5859375, 465.5859375)
3: (-301.2575073, 158.7781830, -301.2575073, 158.7781830, -460.0357056, 460.0357056)
4: (-276.2066956, 210.6250763, -276.2066956, 210.6250763, -486.8317566, 486.8317566)
5: (-246.9300537, 191.3716736, -246.9300537, 191.3716736, -438.3016968, 438.3016968)
6: (-236.3738251, 227.5385132, -236.3738251, 227.5385132, -463.9123535, 463.9123535)
7: (-257.5447693, 215.6144562, -257.5447693, 215.6144562, -473.1591797, 473.1591797)
8: (-309.6375427, 210.9121857, -309.6375427, 210.9121857, -520.5497437, 520.5497437)
9: (-234.0735016, 229.8993225, -234.0735016, 229.8993225, -463.9727783, 463.9727783)

## BASE Result
execution time: IAR + LP analysis = 1.24 + 13.01 = 14.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -460.4076444, upper bound: 460.4076444


# Binary Search by BASE starts (time budget: 2685.75 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=463.9727783203125
rel_dist={9: [-460.40761108255293, 460.4076110701256]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=463.9727783203125
rel_dist={9: [-460.40755870218874, 460.4075586754011]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=463.9727783203125
rel_dist={9: [-460.40746283482497, 460.4074627920745]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=463.9727783203125
rel_dist={9: [-460.4075190791105, 460.4075190850241]}

## Binary Search Result
Binary search time: 71.42 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Individual Split (IS_dual) starts
Time budget: 2614.33 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075582, upper bound: 460.4075137
time: 9.58 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076172, upper bound: 460.4076171
time: 11.23 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 20.94 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 20.94
Output dim: 9, lower bound: -460.4075582, upper bound: 460.4075137
IS_A2, status: Status.UNKNOWN, split count: 1, time: 20.94
Output dim: 9, lower bound: -460.4076172, upper bound: 460.4076171

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -257.6782837, 204.4403687, -256.1317749, 203.2375488, -460.9158020, 460.5721130
1: -216.3702087, 181.7500153, -215.0543060, 180.7030945, -397.0733032, 396.8042603
2: -283.8812256, 183.4134674, -282.1703186, 182.3562164, -466.2374268, 465.5837708
3: -302.2922668, 159.2859650, -300.5665283, 158.4150085, -460.7072754, 459.8524780
4: -277.2501831, 211.4267273, -275.5784607, 210.1466675, -487.3968201, 487.0051880
5: -247.8487549, 192.0652771, -246.3678741, 190.9372559, -438.7859802, 438.4331360
6: -237.2574768, 228.3802948, -235.8358612, 227.0210114, -464.2784424, 464.2161560
7: -258.5010986, 216.3852234, -256.9586182, 215.1226501, -473.6237183, 473.3438416
8: -310.8147888, 211.7235718, -308.9349670, 210.4346161, -521.2493896, 520.6585693
9: -234.9131165, 230.7233582, -233.5390930, 229.3749084, -464.2880249, 464.2624512

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4066719, upper bound: 460.4068207
time: 9.99 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075582, upper bound: 460.4075137
time: 12.41 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -255.4521637, 202.6978302, -256.7159119, 203.7015533, -459.1537170, 459.4137268
1: -214.4840546, 180.2209625, -215.5436707, 181.1157990, -395.5998535, 395.7646179
2: -281.4212341, 181.8703918, -282.8133240, 182.7726288, -464.1938477, 464.6836853
3: -299.7578735, 157.9887543, -301.2575073, 158.7781830, -458.5360718, 459.2462769
4: -274.8458252, 209.5892334, -276.2066956, 210.6250763, -485.4708557, 485.7959290
5: -245.7137909, 190.4309998, -246.9300537, 191.3716736, -437.0854492, 437.3609619
6: -235.2090607, 226.4174042, -236.3738251, 227.5385132, -462.7475586, 462.7912292
7: -256.2756653, 214.5492249, -257.5447693, 215.6144562, -471.8901062, 472.0939941
8: -308.1177063, 209.8820190, -309.6375427, 210.9121857, -519.0299072, 519.5195312
9: -232.9168091, 228.7620850, -234.0735016, 229.8993225, -462.8160706, 462.8355713

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4068465, upper bound: 460.4071141
time: 10.96 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076171, upper bound: 460.4076172
time: 11.45 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.70 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 23.70
Output dim: 9, lower bound: -460.4066719, upper bound: 460.4068207
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.70
Output dim: 9, lower bound: -460.4075582, upper bound: 460.4075137
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 23.70
Output dim: 9, lower bound: -460.4068465, upper bound: 460.4071141
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.70
Output dim: 9, lower bound: -460.4076171, upper bound: 460.4076172

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -257.6782837, 204.4403687, -256.0185547, 203.1482086, -460.8264160, 460.4588928
1: -216.3702087, 181.7500153, -214.9598694, 180.6238098, -396.9940186, 396.7098389
2: -283.8812256, 183.4134674, -282.0462036, 182.2763519, -466.1575928, 465.4596252
3: -302.2922668, 159.2859650, -300.4336853, 158.3460083, -460.6382751, 459.7196655
4: -277.2501831, 211.4267273, -275.4570923, 210.0542450, -487.3044434, 486.8838196
5: -247.8487549, 192.0652771, -246.2592010, 190.8530579, -438.7018127, 438.3244629
6: -237.2574768, 228.3802948, -235.7322540, 226.9213562, -464.1788025, 464.1125183
7: -258.5010986, 216.3852234, -256.8455811, 215.0280457, -473.5291443, 473.2308044
8: -310.8147888, 211.7235718, -308.7998657, 210.3427734, -521.1575928, 520.5234375
9: -234.9131165, 230.7233582, -233.4362183, 229.2741089, -464.1871643, 464.1595459

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.3999153, upper bound: 460.4008803
time: 10.86 seconds

## Relational analysis of IS_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4070738, upper bound: 460.4071064
time: 10.87 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4065464, upper bound: 460.4064160
time: 9.80 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -255.4521637, 202.6978302, -256.6027832, 203.6122437, -459.0643921, 459.3005981
1: -214.4840546, 180.2209625, -215.4492950, 181.0365143, -395.5205383, 395.6702576
2: -281.4212341, 181.8703918, -282.6893616, 182.6928101, -464.1140137, 464.5597534
3: -299.7578735, 157.9887543, -301.1247253, 158.7092438, -458.4670715, 459.1134644
4: -274.8458252, 209.5892334, -276.0853271, 210.5327301, -485.3785095, 485.6745605
5: -245.7137909, 190.4309998, -246.8214417, 191.2875671, -437.0013123, 437.2524109
6: -235.2090607, 226.4174042, -236.2702332, 227.4389343, -462.6479797, 462.6876221
7: -256.2756653, 214.5492249, -257.4317627, 215.5199280, -471.7955933, 471.9809875
8: -308.1177063, 209.8820190, -309.5025330, 210.8203888, -518.9381104, 519.3844604
9: -232.9168091, 228.7620850, -233.9706726, 229.7985535, -462.7153320, 462.7327576

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4016283, upper bound: 460.4005164
time: 9.82 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.3952420, upper bound: 460.3952420
time: 7.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.71 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 18.71
Output dim: 9, lower bound: -460.4070738, upper bound: 460.4071064
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 18.71
Output dim: 9, lower bound: -460.4065464, upper bound: 460.4064160
IS_A2_B2_B1, status: Status.VERIFIED, split count: 3, time: 18.71
Output dim: 9, lower bound: -460.4016283, upper bound: 460.4005164
IS_A2_B2_B2, status: Status.VERIFIED, split count: 3, time: 18.71
Output dim: 9, lower bound: -460.3952420, upper bound: 460.3952420
Binary search (step 0): status=Status.VERIFIED, k_low=2, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=463.9727783203125
rel_dist={9: [-460.4076172501469, 460.40761719415866]}

## Binary search (step 1) starts
Candidate k: 10, corresponding eps: 0.0390625


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075830, upper bound: 460.4075346
time: 9.37 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076337, upper bound: 460.4076337
time: 9.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.47 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.47
Output dim: 9, lower bound: -460.4075830, upper bound: 460.4075346
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.47
Output dim: 9, lower bound: -460.4076337, upper bound: 460.4076337

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -257.6782837, 204.4403687, -256.6597900, 203.6569519, -461.3352356, 461.1001282
1: -216.3702087, 181.7500153, -215.4966278, 181.0761414, -397.4463501, 397.2465820
2: -283.8812256, 183.4134674, -282.7514954, 182.7325897, -466.6138306, 466.1649475
3: -302.2922668, 159.2859650, -301.1911316, 158.7432709, -461.0355225, 460.4771118
4: -277.2501831, 211.4267273, -276.1463013, 210.5790863, -487.8292236, 487.5730286
5: -247.8487549, 192.0652771, -246.8760223, 191.3299408, -439.1786804, 438.9412842
6: -237.2574768, 228.3802948, -236.3220825, 227.4887695, -464.7462463, 464.7023621
7: -258.5010986, 216.3852234, -257.4884338, 215.5671844, -474.0682983, 473.8736572
8: -310.8147888, 211.7235718, -309.5700378, 210.8663025, -521.6810913, 521.2935181
9: -234.9131165, 230.7233582, -234.0221558, 229.8489380, -464.7620239, 464.7454834

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4068187, upper bound: 460.4070727
time: 9.19 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075830, upper bound: 460.4075345
time: 9.60 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -255.4521637, 202.6978302, -256.7159119, 203.7015533, -459.1537170, 459.4137268
1: -214.4840546, 180.2209625, -215.5436707, 181.1157990, -395.5998535, 395.7646179
2: -281.4212341, 181.8703918, -282.8133240, 182.7726288, -464.1938477, 464.6836853
3: -299.7578735, 157.9887543, -301.2575073, 158.7781830, -458.5360718, 459.2462769
4: -274.8458252, 209.5892334, -276.2066956, 210.6250763, -485.4708557, 485.7959290
5: -245.7137909, 190.4309998, -246.9300537, 191.3716736, -437.0854492, 437.3609619
6: -235.2090607, 226.4174042, -236.3738251, 227.5385132, -462.7475586, 462.7912292
7: -256.2756653, 214.5492249, -257.5447693, 215.6144562, -471.8901062, 472.0939941
8: -308.1177063, 209.8820190, -309.6375427, 210.9121857, -519.0299072, 519.5195312
9: -232.9168091, 228.7620850, -234.0735016, 229.8993225, -462.8160706, 462.8355713

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4069081, upper bound: 460.4072358
time: 9.24 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076336, upper bound: 460.4076337
time: 9.45 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.97 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 19.97
Output dim: 9, lower bound: -460.4068187, upper bound: 460.4070727
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 19.97
Output dim: 9, lower bound: -460.4075830, upper bound: 460.4075345
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 19.97
Output dim: 9, lower bound: -460.4069081, upper bound: 460.4072358
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 19.97
Output dim: 9, lower bound: -460.4076336, upper bound: 460.4076337

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -257.6782837, 204.4403687, -256.5466003, 203.5676422, -461.2459106, 460.9869385
1: -216.3702087, 181.7500153, -215.4022522, 180.9968567, -397.3670654, 397.1522217
2: -283.8812256, 183.4134674, -282.6275330, 182.6527710, -466.5339966, 466.0409851
3: -302.2922668, 159.2859650, -301.0583191, 158.6743011, -460.9665527, 460.3442993
4: -277.2501831, 211.4267273, -276.0249939, 210.4867249, -487.7368774, 487.4517212
5: -247.8487549, 192.0652771, -246.7673798, 191.2457886, -439.0945435, 438.8326416
6: -237.2574768, 228.3802948, -236.2185516, 227.3891907, -464.6466370, 464.5988159
7: -258.5010986, 216.3852234, -257.3753967, 215.4726562, -473.9737244, 473.7606201
8: -310.8147888, 211.7235718, -309.4349670, 210.7744904, -521.5892944, 521.1585693
9: -234.9131165, 230.7233582, -233.9192810, 229.7481537, -464.6612244, 464.6426392

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4063660, upper bound: 460.4074428
time: 10.86 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4073594, upper bound: 460.4072988
time: 10.62 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -255.4521637, 202.6978302, -256.6027832, 203.6122437, -459.0643921, 459.3005981
1: -214.4840546, 180.2209625, -215.4492950, 181.0365143, -395.5205383, 395.6702576
2: -281.4212341, 181.8703918, -282.6893616, 182.6928101, -464.1140137, 464.5597534
3: -299.7578735, 157.9887543, -301.1247253, 158.7092438, -458.4670715, 459.1134644
4: -274.8458252, 209.5892334, -276.0853271, 210.5327301, -485.3785095, 485.6745605
5: -245.7137909, 190.4309998, -246.8214417, 191.2875671, -437.0013123, 437.2524109
6: -235.2090607, 226.4174042, -236.2702332, 227.4389343, -462.6479797, 462.6876221
7: -256.2756653, 214.5492249, -257.4317627, 215.5199280, -471.7955933, 471.9809875
8: -308.1177063, 209.8820190, -309.5025330, 210.8203888, -518.9381104, 519.3844604
9: -232.9168091, 228.7620850, -233.9706726, 229.7985535, -462.7153320, 462.7327576

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075472, upper bound: 460.4074941
time: 10.19 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4064171, upper bound: 460.4074410
time: 10.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.94 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 21.94
Output dim: 9, lower bound: -460.4063660, upper bound: 460.4074428
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 21.94
Output dim: 9, lower bound: -460.4073594, upper bound: 460.4072988
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 21.94
Output dim: 9, lower bound: -460.4075472, upper bound: 460.4074941
IS_A2_B2_B2, status: Status.VERIFIED, split count: 3, time: 21.94
Output dim: 9, lower bound: -460.4064171, upper bound: 460.4074410

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -255.4521637, 202.6978302, -254.4508057, 201.9091797, -457.3613281, 457.1486206
1: -214.4840546, 180.2209625, -213.6498413, 179.5298462, -394.0138855, 393.8707886
2: -281.4212341, 181.8703918, -280.3189087, 181.1645355, -462.5857544, 462.1893005
3: -299.7578735, 157.9887543, -298.6065674, 157.3951416, -457.1530151, 456.5953369
4: -274.8458252, 209.5892334, -273.7687378, 208.7640381, -483.6098633, 483.3579712
5: -245.7137909, 190.4309998, -244.7631531, 189.6863556, -435.4001465, 435.1941528
6: -235.2090607, 226.4174042, -234.2995300, 225.5337219, -460.7427368, 460.7169189
7: -256.2756653, 214.5492249, -255.2753448, 213.7144623, -469.9901123, 469.8245544
8: -308.1177063, 209.8820190, -306.9122009, 209.0516663, -517.1693726, 516.7941284
9: -232.9168091, 228.7620850, -232.0014954, 227.8677521, -460.7845154, 460.7635803

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4004989, upper bound: 460.4016692
time: 10.23 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.3866498, upper bound: 460.3852361
time: 8.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 19.55 seconds
IS_A2_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 19.55
Output dim: 9, lower bound: -460.4004989, upper bound: 460.4016692
IS_A2_B2_B1_A2, status: Status.VERIFIED, split count: 4, time: 19.55
Output dim: 9, lower bound: -460.3866498, upper bound: 460.3852361
Binary search (step 1): status=Status.VERIFIED, k_low=8, k_high=12, k_mid=10, eps_mid=0.0390625, abs_max=463.9727783203125
rel_dist={9: [-460.4076338009184, 460.40763368826947]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075899, upper bound: 460.4075410
time: 10.04 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076391, upper bound: 460.4076391
time: 10.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 20.55 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 20.55
Output dim: 9, lower bound: -460.4075899, upper bound: 460.4075410
IS_A2, status: Status.UNKNOWN, split count: 1, time: 20.55
Output dim: 9, lower bound: -460.4076391, upper bound: 460.4076391

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -257.6782837, 204.4403687, -256.7159119, 203.7015533, -461.3798218, 461.1562500
1: -216.3702087, 181.7500153, -215.5436707, 181.1157990, -397.4860229, 397.2936401
2: -283.8812256, 183.4134674, -282.8133240, 182.7726288, -466.6538391, 466.2267456
3: -302.2922668, 159.2859650, -301.2575073, 158.7781830, -461.0704346, 460.5434570
4: -277.2501831, 211.4267273, -276.2066956, 210.6250763, -487.8752136, 487.6334229
5: -247.8487549, 192.0652771, -246.9300537, 191.3716736, -439.2204285, 438.9953003
6: -237.2574768, 228.3802948, -236.3738251, 227.5385132, -464.7959900, 464.7541199
7: -258.5010986, 216.3852234, -257.5447693, 215.6144562, -474.1155090, 473.9299927
8: -310.8147888, 211.7235718, -309.6375427, 210.9121857, -521.7269897, 521.3610840
9: -234.9131165, 230.7233582, -234.0735016, 229.8993225, -464.8123779, 464.7968750

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4068535, upper bound: 460.4071306
time: 9.40 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075899, upper bound: 460.4075409
time: 10.74 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -255.4521637, 202.6978302, -256.7159119, 203.7015533, -459.1537170, 459.4137268
1: -214.4840546, 180.2209625, -215.5436707, 181.1157990, -395.5998535, 395.7646179
2: -281.4212341, 181.8703918, -282.8133240, 182.7726288, -464.1938477, 464.6836853
3: -299.7578735, 157.9887543, -301.2575073, 158.7781830, -458.5360718, 459.2462769
4: -274.8458252, 209.5892334, -276.2066956, 210.6250763, -485.4708557, 485.7959290
5: -245.7137909, 190.4309998, -246.9300537, 191.3716736, -437.0854492, 437.3609619
6: -235.2090607, 226.4174042, -236.3738251, 227.5385132, -462.7475586, 462.7912292
7: -256.2756653, 214.5492249, -257.5447693, 215.6144562, -471.8901062, 472.0939941
8: -308.1177063, 209.8820190, -309.6375427, 210.9121857, -519.0299072, 519.5195312
9: -232.9168091, 228.7620850, -234.0735016, 229.8993225, -462.8160706, 462.8355713

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4069274, upper bound: 460.4072682
time: 9.39 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076390, upper bound: 460.4076390
time: 10.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.58 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 21.58
Output dim: 9, lower bound: -460.4068535, upper bound: 460.4071306
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.58
Output dim: 9, lower bound: -460.4075899, upper bound: 460.4075409
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 21.58
Output dim: 9, lower bound: -460.4069274, upper bound: 460.4072682
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.58
Output dim: 9, lower bound: -460.4076390, upper bound: 460.4076390

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -257.6782837, 204.4403687, -256.6027832, 203.6122437, -461.2905273, 461.0430908
1: -216.3702087, 181.7500153, -215.4492950, 181.0365143, -397.4067383, 397.1992798
2: -283.8812256, 183.4134674, -282.6893616, 182.6928101, -466.5740051, 466.1028442
3: -302.2922668, 159.2859650, -301.1247253, 158.7092438, -461.0014648, 460.4107056
4: -277.2501831, 211.4267273, -276.0853271, 210.5327301, -487.7828674, 487.5120544
5: -247.8487549, 192.0652771, -246.8214417, 191.2875671, -439.1362610, 438.8867188
6: -237.2574768, 228.3802948, -236.2702332, 227.4389343, -464.6964111, 464.6505127
7: -258.5010986, 216.3852234, -257.4317627, 215.5199280, -474.0209961, 473.8169861
8: -310.8147888, 211.7235718, -309.5025330, 210.8203888, -521.6351318, 521.2260132
9: -234.9131165, 230.7233582, -233.9706726, 229.7985535, -464.7116394, 464.6940308

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074485, upper bound: 460.4074555
time: 10.86 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4073666, upper bound: 460.4073048
time: 9.24 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -255.4521637, 202.6978302, -256.6027832, 203.6122437, -459.0643921, 459.3005981
1: -214.4840546, 180.2209625, -215.4492950, 181.0365143, -395.5205383, 395.6702576
2: -281.4212341, 181.8703918, -282.6893616, 182.6928101, -464.1140137, 464.5597534
3: -299.7578735, 157.9887543, -301.1247253, 158.7092438, -458.4670715, 459.1134644
4: -274.8458252, 209.5892334, -276.0853271, 210.5327301, -485.3785095, 485.6745605
5: -245.7137909, 190.4309998, -246.8214417, 191.2875671, -437.0013123, 437.2524109
6: -235.2090607, 226.4174042, -236.2702332, 227.4389343, -462.6479797, 462.6876221
7: -256.2756653, 214.5492249, -257.4317627, 215.5199280, -471.7955933, 471.9809875
8: -308.1177063, 209.8820190, -309.5025330, 210.8203888, -518.9381104, 519.3844604
9: -232.9168091, 228.7620850, -233.9706726, 229.7985535, -462.7153320, 462.7327576

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075575, upper bound: 460.4075011
time: 9.89 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074448, upper bound: 460.4074447
time: 9.09 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.26 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 20.26
Output dim: 9, lower bound: -460.4074485, upper bound: 460.4074555
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 20.26
Output dim: 9, lower bound: -460.4073666, upper bound: 460.4073048
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 20.26
Output dim: 9, lower bound: -460.4075575, upper bound: 460.4075011
IS_A2_B2_B2, status: Status.VERIFIED, split count: 3, time: 20.26
Output dim: 9, lower bound: -460.4074448, upper bound: 460.4074447

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -255.4521637, 202.6978302, -254.4508057, 201.9091797, -457.3613281, 457.1486206
1: -214.4840546, 180.2209625, -213.6498413, 179.5298462, -394.0138855, 393.8707886
2: -281.4212341, 181.8703918, -280.3189087, 181.1645355, -462.5857544, 462.1893005
3: -299.7578735, 157.9887543, -298.6065674, 157.3951416, -457.1530151, 456.5953369
4: -274.8458252, 209.5892334, -273.7687378, 208.7640381, -483.6098633, 483.3579712
5: -245.7137909, 190.4309998, -244.7631531, 189.6863556, -435.4001465, 435.1941528
6: -235.2090607, 226.4174042, -234.2995300, 225.5337219, -460.7427368, 460.7169189
7: -256.2756653, 214.5492249, -255.2753448, 213.7144623, -469.9901123, 469.8245544
8: -308.1177063, 209.8820190, -306.9122009, 209.0516663, -517.1693726, 516.7941284
9: -232.9168091, 228.7620850, -232.0014954, 227.8677521, -460.7845154, 460.7635803

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4005775, upper bound: 460.4017777
time: 9.07 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.3870963, upper bound: 460.3856062
time: 6.72 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 17.08 seconds
IS_A2_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 17.08
Output dim: 9, lower bound: -460.4005775, upper bound: 460.4017777
IS_A2_B2_B1_A2, status: Status.VERIFIED, split count: 4, time: 17.08
Output dim: 9, lower bound: -460.3870963, upper bound: 460.3856062
Binary search (step 2): status=Status.VERIFIED, k_low=11, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=463.9727783203125
rel_dist={9: [-460.40763907635636, 460.4076391076393]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075964, upper bound: 460.4075474
time: 7.74 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076444, upper bound: 460.4076443
time: 8.92 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.79
Output dim: 9, lower bound: -460.4075964, upper bound: 460.4075474
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.79
Output dim: 9, lower bound: -460.4076444, upper bound: 460.4076443

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -257.6782837, 204.4403687, -256.7159119, 203.7015533, -461.3798218, 461.1562500
1: -216.3702087, 181.7500153, -215.5436707, 181.1157990, -397.4860229, 397.2936401
2: -283.8812256, 183.4134674, -282.8133240, 182.7726288, -466.6538391, 466.2267456
3: -302.2922668, 159.2859650, -301.2575073, 158.7781830, -461.0704346, 460.5434570
4: -277.2501831, 211.4267273, -276.2066956, 210.6250763, -487.8752136, 487.6334229
5: -247.8487549, 192.0652771, -246.9300537, 191.3716736, -439.2204285, 438.9953003
6: -237.2574768, 228.3802948, -236.3738251, 227.5385132, -464.7959900, 464.7541199
7: -258.5010986, 216.3852234, -257.5447693, 215.6144562, -474.1155090, 473.9299927
8: -310.8147888, 211.7235718, -309.6375427, 210.9121857, -521.7269897, 521.3610840
9: -234.9131165, 230.7233582, -234.0735016, 229.8993225, -464.8123779, 464.7968750

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 174

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074573, upper bound: 460.4074677
time: 8.09 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4073735, upper bound: 460.4073106
time: 9.32 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -255.4521637, 202.6978302, -256.7159119, 203.7015533, -459.1537170, 459.4137268
1: -214.4840546, 180.2209625, -215.5436707, 181.1157990, -395.5998535, 395.7646179
2: -281.4212341, 181.8703918, -282.8133240, 182.7726288, -464.1938477, 464.6836853
3: -299.7578735, 157.9887543, -301.2575073, 158.7781830, -458.5360718, 459.2462769
4: -274.8458252, 209.5892334, -276.2066956, 210.6250763, -485.4708557, 485.7959290
5: -245.7137909, 190.4309998, -246.9300537, 191.3716736, -437.0854492, 437.3609619
6: -235.2090607, 226.4174042, -236.3738251, 227.5385132, -462.7475586, 462.7912292
7: -256.2756653, 214.5492249, -257.5447693, 215.6144562, -471.8901062, 472.0939941
8: -308.1177063, 209.8820190, -309.6375427, 210.9121857, -519.0299072, 519.5195312
9: -232.9168091, 228.7620850, -234.0735016, 229.8993225, -462.8160706, 462.8355713

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075674, upper bound: 460.4075083
time: 9.31 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074485, upper bound: 460.4074484
time: 8.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.15 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 19.15
Output dim: 9, lower bound: -460.4074573, upper bound: 460.4074677
IS_A1_A2, status: Status.VERIFIED, split count: 2, time: 19.15
Output dim: 9, lower bound: -460.4073735, upper bound: 460.4073106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 19.15
Output dim: 9, lower bound: -460.4075674, upper bound: 460.4075083
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 19.15
Output dim: 9, lower bound: -460.4074485, upper bound: 460.4074484

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -255.4521637, 202.6978302, -254.5633850, 201.9980011, -457.4501648, 457.2612305
1: -214.4840546, 180.2209625, -213.7437134, 179.6086884, -394.0927429, 393.9646606
2: -281.4212341, 181.8703918, -280.4422302, 181.2439423, -462.6651611, 462.3125916
3: -299.7578735, 157.9887543, -298.7386169, 157.4637756, -457.2216492, 456.7273560
4: -274.8458252, 209.5892334, -273.8894653, 208.8558960, -483.7017212, 483.4786987
5: -245.7137909, 190.4309998, -244.8712158, 189.7700348, -435.4838257, 435.3021851
6: -235.2090607, 226.4174042, -234.4025269, 225.6327972, -460.8417969, 460.8199463
7: -256.2756653, 214.5492249, -255.3877258, 213.8085022, -470.0841675, 469.9369202
8: -308.1177063, 209.8820190, -307.0464783, 209.1429901, -517.2606812, 516.9284668
9: -232.9168091, 228.7620850, -232.1037598, 227.9679718, -460.8847351, 460.8658447

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4069791, upper bound: 460.4065187
time: 7.25 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075673, upper bound: 460.4075082
time: 8.12 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 16.65 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 16.65
Output dim: 9, lower bound: -460.4069791, upper bound: 460.4065187
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.65
Output dim: 9, lower bound: -460.4075673, upper bound: 460.4075082

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -255.3390198, 202.6085663, -254.5633850, 201.9980011, -457.3370361, 457.1719360
1: -214.3897095, 180.1417084, -213.7437134, 179.6086884, -393.9984131, 393.8854065
2: -281.2973328, 181.7906036, -280.4422302, 181.2439423, -462.5411987, 462.2328186
3: -299.6251831, 157.9197845, -298.7386169, 157.4637756, -457.0889587, 456.6583862
4: -274.7245483, 209.4969177, -273.8894653, 208.8558960, -483.5804443, 483.3863525
5: -245.6052094, 190.3468781, -244.8712158, 189.7700348, -435.3752136, 435.2180786
6: -235.1055603, 226.3178864, -234.4025269, 225.6327972, -460.7382812, 460.7203979
7: -256.1627808, 214.4547577, -255.3877258, 213.8085022, -469.9712524, 469.8424072
8: -307.9826660, 209.7902527, -307.0464783, 209.1429901, -517.1256104, 516.8367310
9: -232.8140564, 228.6613617, -232.1037598, 227.9679718, -460.7819824, 460.7651367

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4006401, upper bound: 460.4018754
time: 8.27 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.3875393, upper bound: 460.3859399
time: 7.05 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 16.60 seconds
IS_A2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 16.60
Output dim: 9, lower bound: -460.4006401, upper bound: 460.4018754
IS_A2_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 16.60
Output dim: 9, lower bound: -460.3875393, upper bound: 460.3859399
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=463.9727783203125
rel_dist={9: [-460.40764437973144, 460.40764437529225]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 519.25 seconds
