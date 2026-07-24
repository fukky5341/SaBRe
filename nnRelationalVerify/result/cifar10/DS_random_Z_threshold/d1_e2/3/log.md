## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 3)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.005602400400000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1473250, 0.1473249)
1: (0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170227, 0.0170227)
2: (-3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0895383, 0.0895383)
3: (-3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1919439, 0.1919439)
4: (-3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1612116, 0.1612116)
5: (-3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1990502, 0.1990501)
6: (-3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1346494, 0.1346494)
7: (-0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1157507, 0.1157507)
8: (-1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502335, 0.0502335)
9: (-1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451960, 0.0451960)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.80 + 55.83 = 63.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0056239, upper bound: 0.0056236

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2415

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2499

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056212, upper bound: 0.0056233
time: 76.02 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056225, upper bound: 0.0056222
time: 33.09 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 109.13 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 109.13
Output dim: 1, lower bound: -0.0056212, upper bound: 0.0056233
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 109.13
Output dim: 1, lower bound: -0.0056225, upper bound: 0.0056222

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472872, 0.1472840
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170132, 0.0170140
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0894538, 0.0894562
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1919405, 0.1919401
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1611745, 0.1611702
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1990463, 0.1990459
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1346060, 0.1346022
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1157455, 0.1157453
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502291, 0.0502286
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451401, 0.0451457

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2380

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056210, upper bound: 0.0056173
time: 8.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056148, upper bound: 0.0056241
time: 4.59 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472840, 0.1472872
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170140, 0.0170132
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0894562, 0.0894538
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1919401, 0.1919405
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1611701, 0.1611745
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1990459, 0.1990463
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1346022, 0.1346060
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1157453, 0.1157455
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502286, 0.0502291
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451457, 0.0451401

Time for backsubstitution: 6.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2529

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2235

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056227, upper bound: 0.0056226
time: 51.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056227, upper bound: 0.0056218
time: 70.39 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 128.08 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 128.08
Output dim: 1, lower bound: -0.0056210, upper bound: 0.0056173
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 128.08
Output dim: 1, lower bound: -0.0056148, upper bound: 0.0056241
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 128.08
Output dim: 1, lower bound: -0.0056227, upper bound: 0.0056226
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 128.08
Output dim: 1, lower bound: -0.0056227, upper bound: 0.0056218

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472847, 0.1472816
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0169840, 0.0169799
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0891830, 0.0892237
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916025, 0.1916515
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1605976, 0.1606753
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1987541, 0.1987966
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1344735, 0.1344886
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1154609, 0.1155002
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502278, 0.0502272
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451367, 0.0451423

Time for backsubstitution: 6.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 149

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2051

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056204, upper bound: 0.0056164
time: 56.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056208, upper bound: 0.0056166
time: 21.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472849, 0.1472815
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0169791, 0.0169848
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0892212, 0.0891855
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916519, 0.1916022
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1606796, 0.1605932
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1987970, 0.1987537
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1344924, 0.1344697
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1155005, 0.1154607
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502278, 0.0502273
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451367, 0.0451422

Time for backsubstitution: 6.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2936

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056147, upper bound: 0.0056233
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056142, upper bound: 0.0056237
time: 3.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472840, 0.1472872
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170140, 0.0170132
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0894562, 0.0894538
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1919401, 0.1919405
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1611701, 0.1611745
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1990459, 0.1990463
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1346022, 0.1346060
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1157453, 0.1157455
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502286, 0.0502291
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451457, 0.0451401

Time for backsubstitution: 6.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2702

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2952

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056223, upper bound: 0.0056220
time: 22.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056222, upper bound: 0.0056219
time: 10.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472840, 0.1472872
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170140, 0.0170132
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0894562, 0.0894538
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1919401, 0.1919405
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1611701, 0.1611745
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1990459, 0.1990463
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1346022, 0.1346060
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1157453, 0.1157455
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502286, 0.0502291
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451457, 0.0451401

Time for backsubstitution: 6.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2271

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2926

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056224, upper bound: 0.0056217
time: 62.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056223, upper bound: 0.0056221
time: 37.41 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 106.64 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 106.64
Output dim: 1, lower bound: -0.0056204, upper bound: 0.0056164
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 106.64
Output dim: 1, lower bound: -0.0056208, upper bound: 0.0056166
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 106.64
Output dim: 1, lower bound: -0.0056147, upper bound: 0.0056233
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 106.64
Output dim: 1, lower bound: -0.0056142, upper bound: 0.0056237
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 106.64
Output dim: 1, lower bound: -0.0056223, upper bound: 0.0056220
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 106.64
Output dim: 1, lower bound: -0.0056222, upper bound: 0.0056219
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 106.64
Output dim: 1, lower bound: -0.0056224, upper bound: 0.0056217
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 106.64
Output dim: 1, lower bound: -0.0056223, upper bound: 0.0056221

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472811, 0.1472767
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0169828, 0.0169790
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0891731, 0.0892069
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1915998, 0.1916488
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1605959, 0.1606718
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1987502, 0.1987927
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1344565, 0.1344733
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1154563, 0.1154962
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502244, 0.0502231
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451104, 0.0451184

Time for backsubstitution: 6.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2650

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 491

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056203, upper bound: 0.0055921
time: 6.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0055967, upper bound: 0.0056168
time: 36.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472798, 0.1472780
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0169830, 0.0169787
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0891662, 0.0892137
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1915998, 0.1916488
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1605942, 0.1606736
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1987502, 0.1987928
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1344581, 0.1344717
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1154569, 0.1154957
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502237, 0.0502238
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451128, 0.0451161

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2940

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2236

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056212, upper bound: 0.0056164
time: 6.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056212, upper bound: 0.0056164
time: 4.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472783, 0.1472750
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0169733, 0.0169788
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0891968, 0.0891622
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916497, 0.1916000
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1606595, 0.1605736
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1987895, 0.1987461
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1344894, 0.1344667
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1154901, 0.1154500
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502112, 0.0502109
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451339, 0.0451393

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2407

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2743

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056147, upper bound: 0.0056155
time: 55.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056142, upper bound: 0.0056224
time: 26.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472785, 0.1472749
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0169731, 0.0169790
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0891980, 0.0891611
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916498, 0.1915999
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1606599, 0.1605732
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1987895, 0.1987461
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1344894, 0.1344667
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1154898, 0.1154504
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502114, 0.0502107
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451338, 0.0451394

Time for backsubstitution: 6.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1096

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056143, upper bound: 0.0056234
time: 40.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056143, upper bound: 0.0056237
time: 4.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472590, 0.1472616
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170136, 0.0170127
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0893507, 0.0893470
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1919317, 0.1919321
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1611495, 0.1611543
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1990367, 0.1990371
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1345180, 0.1345229
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1157333, 0.1157323
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502000, 0.0502000
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0449830, 0.0449902

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2051

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2723

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056220, upper bound: 0.0056194
time: 417.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056211, upper bound: 0.0056212
time: 4.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472584, 0.1472622
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170136, 0.0170128
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0893495, 0.0893482
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1919317, 0.1919321
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1611499, 0.1611539
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1990367, 0.1990371
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1345191, 0.1345217
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1157321, 0.1157335
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0501995, 0.0502006
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0449957, 0.0449774

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 754

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056183, upper bound: 0.0056188
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056181, upper bound: 0.0056180
time: 20.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472753, 0.1472804
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170139, 0.0170131
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0894434, 0.0894417
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1919404, 0.1919408
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1611683, 0.1611727
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1990471, 0.1990475
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1346031, 0.1346068
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1157447, 0.1157449
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502278, 0.0502282
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451462, 0.0451406

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2927

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2346

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056171, upper bound: 0.0055930
time: 110.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0055938, upper bound: 0.0056169
time: 3.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472771, 0.1472786
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170139, 0.0170131
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0894441, 0.0894409
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1919404, 0.1919408
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1611683, 0.1611726
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1990471, 0.1990475
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1346031, 0.1346068
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1157447, 0.1157449
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502276, 0.0502283
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451461, 0.0451406

Time for backsubstitution: 6.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2696

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056225, upper bound: 0.0056156
time: 11.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056157, upper bound: 0.0056218
time: 9.37 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 27.00 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.00
Output dim: 1, lower bound: -0.0056203, upper bound: 0.0055921
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.00
Output dim: 1, lower bound: -0.0055967, upper bound: 0.0056168
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.00
Output dim: 1, lower bound: -0.0056212, upper bound: 0.0056164
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.00
Output dim: 1, lower bound: -0.0056212, upper bound: 0.0056164
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.00
Output dim: 1, lower bound: -0.0056147, upper bound: 0.0056155
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.00
Output dim: 1, lower bound: -0.0056142, upper bound: 0.0056224
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.00
Output dim: 1, lower bound: -0.0056143, upper bound: 0.0056234
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.00
Output dim: 1, lower bound: -0.0056143, upper bound: 0.0056237
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.00
Output dim: 1, lower bound: -0.0056220, upper bound: 0.0056194
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.00
Output dim: 1, lower bound: -0.0056211, upper bound: 0.0056212
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.00
Output dim: 1, lower bound: -0.0056183, upper bound: 0.0056188
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.00
Output dim: 1, lower bound: -0.0056181, upper bound: 0.0056180
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.00
Output dim: 1, lower bound: -0.0056171, upper bound: 0.0055930
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.00
Output dim: 1, lower bound: -0.0055938, upper bound: 0.0056169
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.00
Output dim: 1, lower bound: -0.0056225, upper bound: 0.0056156
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.00
Output dim: 1, lower bound: -0.0056157, upper bound: 0.0056218

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1471564, 0.1471640
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0169271, 0.0169159
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0891455, 0.0891790
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1915456, 0.1915883
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1601917, 0.1603131
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1986375, 0.1986646
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1344501, 0.1344667
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1153912, 0.1154247
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0500720, 0.0500901
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451104, 0.0451184

Time for backsubstitution: 6.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2729

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1115

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056203, upper bound: 0.0055923
time: 51.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056203, upper bound: 0.0055927
time: 49.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1471689, 0.1471519
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0169196, 0.0169233
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0891452, 0.0891794
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1915393, 0.1915947
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1602372, 0.1602676
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1986221, 0.1986800
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1344500, 0.1344669
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1153847, 0.1154314
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0500913, 0.0500707
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451104, 0.0451184

Time for backsubstitution: 6.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2630

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2926

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0055966, upper bound: 0.0056172
time: 3.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0055966, upper bound: 0.0056160
time: 28.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472798, 0.1472780
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0169830, 0.0169787
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0891662, 0.0892137
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1915998, 0.1916488
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1605942, 0.1606736
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1987502, 0.1987928
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1344581, 0.1344717
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1154569, 0.1154957
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502237, 0.0502238
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451128, 0.0451161

Time for backsubstitution: 6.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 754

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056207, upper bound: 0.0056145
time: 9.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056193, upper bound: 0.0056146
time: 49.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472798, 0.1472780
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0169830, 0.0169787
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0891662, 0.0892137
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1915998, 0.1916488
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1605942, 0.1606736
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1987502, 0.1987928
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1344581, 0.1344717
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1154569, 0.1154957
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502237, 0.0502238
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451128, 0.0451161

Time for backsubstitution: 6.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2938

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056208, upper bound: 0.0056167
time: 26.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056208, upper bound: 0.0056167
time: 26.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472754, 0.1472723
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0169541, 0.0169592
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0890927, 0.0890617
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1914690, 0.1914189
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1605130, 0.1604272
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1986100, 0.1985653
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1340979, 0.1340785
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1154316, 0.1153907
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0501005, 0.0501006
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451340, 0.0451394

Time for backsubstitution: 6.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2231

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3111

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056127, upper bound: 0.0056170
time: 7.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056102, upper bound: 0.0056196
time: 10.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472754, 0.1472723
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0169537, 0.0169596
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0890963, 0.0890581
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1914687, 0.1914192
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1605130, 0.1604272
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1986087, 0.1985666
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1341013, 0.1340751
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1154309, 0.1153915
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0501009, 0.0501002
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451340, 0.0451394

Time for backsubstitution: 6.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2449

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056143, upper bound: 0.0056227
time: 23.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056143, upper bound: 0.0056228
time: 4.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472784, 0.1472748
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0169731, 0.0169790
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0891980, 0.0891611
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916497, 0.1915999
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1606600, 0.1605732
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1987895, 0.1987461
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1344894, 0.1344667
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1154898, 0.1154504
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502114, 0.0502107
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451337, 0.0451394

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 1117

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 711

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056137, upper bound: 0.0056212
time: 16.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056121, upper bound: 0.0056232
time: 3.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472784, 0.1472748
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0169731, 0.0169790
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0891980, 0.0891611
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916497, 0.1915999
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1606600, 0.1605732
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1987895, 0.1987461
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1344894, 0.1344667
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1154898, 0.1154504
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502114, 0.0502107
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451337, 0.0451394

Time for backsubstitution: 6.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2723

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056144, upper bound: 0.0056242
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056144, upper bound: 0.0056242
time: 3.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472154, 0.1472178
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0169887, 0.0169865
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0892710, 0.0892725
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1918777, 0.1918812
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1610185, 0.1610314
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989687, 0.1989723
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1342376, 0.1342516
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1156144, 0.1156049
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0501066, 0.0501008
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0448288, 0.0448369

Time for backsubstitution: 6.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2075

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2277

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056218, upper bound: 0.0056168
time: 4.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056181, upper bound: 0.0056203
time: 34.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1472150, 0.1472180
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0169874, 0.0169880
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0892747, 0.0892673
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1918809, 0.1918781
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1610266, 0.1610232
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989718, 0.1989692
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1342484, 0.1342425
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1156059, 0.1156133
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0501008, 0.0501066
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0448297, 0.0448360

Time for backsubstitution: 6.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2075

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056168, upper bound: 0.0056172
time: 39.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056175, upper bound: 0.0056162
time: 91.72 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 137.55 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056203, upper bound: 0.0055923
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056203, upper bound: 0.0055927
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0055966, upper bound: 0.0056172
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0055966, upper bound: 0.0056160
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056207, upper bound: 0.0056145
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056193, upper bound: 0.0056146
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056208, upper bound: 0.0056167
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056208, upper bound: 0.0056167
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056127, upper bound: 0.0056170
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056102, upper bound: 0.0056196
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056143, upper bound: 0.0056227
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056143, upper bound: 0.0056228
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056137, upper bound: 0.0056212
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056121, upper bound: 0.0056232
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056144, upper bound: 0.0056242
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056144, upper bound: 0.0056242
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056218, upper bound: 0.0056168
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056181, upper bound: 0.0056203
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056168, upper bound: 0.0056172
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 137.55
Output dim: 1, lower bound: -0.0056175, upper bound: 0.0056162
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 137.55
Output dim: 1, lower bound: -0.0056183, upper bound: 0.0056188
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 137.55
Output dim: 1, lower bound: -0.0056181, upper bound: 0.0056180
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 137.55
Output dim: 1, lower bound: -0.0056171, upper bound: 0.0055930
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 137.55
Output dim: 1, lower bound: -0.0055938, upper bound: 0.0056169
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 137.55
Output dim: 1, lower bound: -0.0056225, upper bound: 0.0056156
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 137.55
Output dim: 1, lower bound: -0.0056157, upper bound: 0.0056218

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 63.63 + 1860.95 = 1924.58 seconds
