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
execution time: IAR + RelationalAnalysis = 7.94 + 58.90 = 66.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0056239, upper bound: 0.0056236

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 408

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056184, upper bound: 0.0056201
time: 10.28 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056188, upper bound: 0.0056193
time: 3.23 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 13.59 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 13.59
Output dim: 1, lower bound: -0.0056184, upper bound: 0.0056201
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 13.59
Output dim: 1, lower bound: -0.0056188, upper bound: 0.0056193

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1473137, 0.1473140
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170229, 0.0170229
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0895282, 0.0895284
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1919537, 0.1919544
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1611466, 0.1611481
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1990651, 0.1990661
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1346530, 0.1346532
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1156984, 0.1156999
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502226, 0.0502223
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451697, 0.0451702

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3090

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056179, upper bound: 0.0056200
time: 5.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056185, upper bound: 0.0056198
time: 5.64 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1473140, 0.1473137
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170229, 0.0170229
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0895284, 0.0895282
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1919544, 0.1919537
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1611481, 0.1611466
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1990661, 0.1990651
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1346532, 0.1346530
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1156999, 0.1156984
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0502223, 0.0502226
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0451702, 0.0451697

Time for backsubstitution: 6.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3090

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056183, upper bound: 0.0056196
time: 8.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056189, upper bound: 0.0056193
time: 6.74 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 21.30 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 21.30
Output dim: 1, lower bound: -0.0056179, upper bound: 0.0056200
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 21.30
Output dim: 1, lower bound: -0.0056185, upper bound: 0.0056198
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 21.30
Output dim: 1, lower bound: -0.0056183, upper bound: 0.0056196
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 21.30
Output dim: 1, lower bound: -0.0056189, upper bound: 0.0056193

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1470268, 0.1470170
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170155, 0.0170157
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0894397, 0.0894427
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1918648, 0.1918600
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1610309, 0.1610373
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989553, 0.1989496
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1346553, 0.1346553
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1154474, 0.1154577
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0498629, 0.0498523
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0445973, 0.0446125

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3514

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056108, upper bound: 0.0056200
time: 4.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056179, upper bound: 0.0056126
time: 13.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1470166, 0.1470271
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170157, 0.0170155
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0894425, 0.0894399
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1918593, 0.1918655
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1610358, 0.1610324
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989486, 0.1989563
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1346552, 0.1346555
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1154563, 0.1154489
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0498526, 0.0498626
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0446120, 0.0445978

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3514

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056111, upper bound: 0.0056194
time: 8.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056182, upper bound: 0.0056122
time: 91.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1470271, 0.1470167
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170155, 0.0170157
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0894399, 0.0894425
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1918655, 0.1918593
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1610324, 0.1610358
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989562, 0.1989486
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1346555, 0.1346552
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1154489, 0.1154562
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0498626, 0.0498526
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0445978, 0.0446120

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3514

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056112, upper bound: 0.0056111
time: 22.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056182, upper bound: 0.0056122
time: 79.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1470170, 0.1470268
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170157, 0.0170155
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0894427, 0.0894397
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1918600, 0.1918648
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1610373, 0.1610309
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989496, 0.1989552
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1346553, 0.1346553
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1154577, 0.1154474
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0498524, 0.0498629
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0446125, 0.0445972

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3514

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056115, upper bound: 0.0056191
time: 9.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056186, upper bound: 0.0056117
time: 30.89 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 46.60 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 46.60
Output dim: 1, lower bound: -0.0056108, upper bound: 0.0056200
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 46.60
Output dim: 1, lower bound: -0.0056179, upper bound: 0.0056126
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 46.60
Output dim: 1, lower bound: -0.0056111, upper bound: 0.0056194
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 46.60
Output dim: 1, lower bound: -0.0056182, upper bound: 0.0056122
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 46.60
Output dim: 1, lower bound: -0.0056112, upper bound: 0.0056111
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 46.60
Output dim: 1, lower bound: -0.0056182, upper bound: 0.0056122
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 46.60
Output dim: 1, lower bound: -0.0056115, upper bound: 0.0056191
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 46.60
Output dim: 1, lower bound: -0.0056186, upper bound: 0.0056117

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1467988, 0.1467934
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170136, 0.0170138
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0876758, 0.0875860
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1918262, 0.1918211
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1599388, 0.1598878
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1991169, 0.1991117
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1339812, 0.1340137
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1143251, 0.1142879
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0493604, 0.0493748
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0434630, 0.0434192

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2415

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056102, upper bound: 0.0056196
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056108, upper bound: 0.0056188
time: 4.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1468033, 0.1467890
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170136, 0.0170138
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0875829, 0.0876788
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1918260, 0.1918214
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1598814, 0.1599452
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1991174, 0.1991113
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1340137, 0.1339812
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1142775, 0.1143354
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0493854, 0.0493498
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0434039, 0.0434783

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2415

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056173, upper bound: 0.0056125
time: 7.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056178, upper bound: 0.0056121
time: 4.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1467886, 0.1468036
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170138, 0.0170136
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0876786, 0.0875831
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1918207, 0.1918266
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1599438, 0.1598829
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1991103, 0.1991183
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1339810, 0.1340138
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1143339, 0.1142790
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0493501, 0.0493851
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0434778, 0.0434044

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2415

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056106, upper bound: 0.0056193
time: 13.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056111, upper bound: 0.0056185
time: 51.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1467931, 0.1467991
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170138, 0.0170136
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0875858, 0.0876760
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1918204, 0.1918269
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1598863, 0.1599404
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1991107, 0.1991179
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1340135, 0.1339813
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1142864, 0.1143266
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0493751, 0.0493601
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0434187, 0.0434635

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2415

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056176, upper bound: 0.0056122
time: 5.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056182, upper bound: 0.0056116
time: 5.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1467991, 0.1467931
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170136, 0.0170138
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0876760, 0.0875858
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1918269, 0.1918204
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1599403, 0.1598863
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1991179, 0.1991107
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1339813, 0.1340135
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1143266, 0.1142864
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0493601, 0.0493751
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0434635, 0.0434187

Time for backsubstitution: 6.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2415

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056106, upper bound: 0.0056197
time: 4.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056112, upper bound: 0.0056184
time: 6.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1468036, 0.1467886
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170136, 0.0170138
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0875831, 0.0876786
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1918266, 0.1918207
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1598829, 0.1599438
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1991184, 0.1991102
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1340138, 0.1339810
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1142790, 0.1143339
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0493851, 0.0493501
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0434044, 0.0434778

Time for backsubstitution: 6.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 2415

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056176, upper bound: 0.0056121
time: 29.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056182, upper bound: 0.0056113
time: 5.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1467890, 0.1468033
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170138, 0.0170136
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0876788, 0.0875829
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1918214, 0.1918259
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1599452, 0.1598814
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1991113, 0.1991174
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1339812, 0.1340137
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1143354, 0.1142775
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0493498, 0.0493854
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0434783, 0.0434039

Time for backsubstitution: 6.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2415

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056109, upper bound: 0.0056184
time: 34.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056115, upper bound: 0.0056178
time: 50.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1467934, 0.1467988
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170138, 0.0170136
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0875860, 0.0876758
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1918211, 0.1918262
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1598878, 0.1599388
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1991117, 0.1991169
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1340137, 0.1339812
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1142879, 0.1143250
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0493748, 0.0493604
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0434192, 0.0434630

Time for backsubstitution: 6.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2415

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056180, upper bound: 0.0056109
time: 47.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056186, upper bound: 0.0056109
time: 7.40 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 61.15 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 61.15
Output dim: 1, lower bound: -0.0056102, upper bound: 0.0056196
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 61.15
Output dim: 1, lower bound: -0.0056108, upper bound: 0.0056188
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 61.15
Output dim: 1, lower bound: -0.0056173, upper bound: 0.0056125
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 61.15
Output dim: 1, lower bound: -0.0056178, upper bound: 0.0056121
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 61.15
Output dim: 1, lower bound: -0.0056106, upper bound: 0.0056193
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 61.15
Output dim: 1, lower bound: -0.0056111, upper bound: 0.0056185
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 61.15
Output dim: 1, lower bound: -0.0056176, upper bound: 0.0056122
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 61.15
Output dim: 1, lower bound: -0.0056182, upper bound: 0.0056116
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 61.15
Output dim: 1, lower bound: -0.0056106, upper bound: 0.0056197
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 61.15
Output dim: 1, lower bound: -0.0056112, upper bound: 0.0056184
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 61.15
Output dim: 1, lower bound: -0.0056176, upper bound: 0.0056121
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 61.15
Output dim: 1, lower bound: -0.0056182, upper bound: 0.0056113
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 61.15
Output dim: 1, lower bound: -0.0056109, upper bound: 0.0056184
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 61.15
Output dim: 1, lower bound: -0.0056115, upper bound: 0.0056178
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 61.15
Output dim: 1, lower bound: -0.0056180, upper bound: 0.0056109
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 61.15
Output dim: 1, lower bound: -0.0056186, upper bound: 0.0056109

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1463543, 0.1463308
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170020, 0.0170027
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0875354, 0.0874507
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916701, 0.1916565
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1597791, 0.1597340
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989297, 0.1989145
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1339844, 0.1340168
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1139428, 0.1139196
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0487838, 0.0487758
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0425343, 0.0425258

Time for backsubstitution: 6.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056097, upper bound: 0.0056195
time: 5.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056102, upper bound: 0.0056194
time: 3.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1463362, 0.1463490
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170025, 0.0170023
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0875405, 0.0874456
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916617, 0.1916650
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1597850, 0.1597280
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989197, 0.1989244
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1339842, 0.1340169
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1139568, 0.1139056
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0487613, 0.0487984
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0425696, 0.0424905

Time for backsubstitution: 6.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056101, upper bound: 0.0056182
time: 82.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056108, upper bound: 0.0056188
time: 17.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1463588, 0.1463264
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170021, 0.0170027
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0874426, 0.0875436
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916699, 0.1916568
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1597216, 0.1597914
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989301, 0.1989141
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1340169, 0.1339843
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1138953, 0.1139671
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0488088, 0.0487508
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0424752, 0.0425849

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056168, upper bound: 0.0056124
time: 10.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056173, upper bound: 0.0056124
time: 8.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1463407, 0.1463445
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170025, 0.0170023
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0874477, 0.0875385
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916614, 0.1916653
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1597276, 0.1597855
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989202, 0.1989240
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1340168, 0.1339844
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1139092, 0.1139532
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0487863, 0.0487733
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0425105, 0.0425496

Time for backsubstitution: 6.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 2175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056172, upper bound: 0.0056120
time: 30.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056179, upper bound: 0.0056120
time: 4.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1463441, 0.1463410
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170023, 0.0170025
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0875383, 0.0874479
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916646, 0.1916620
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1597840, 0.1597291
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989230, 0.1989211
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1339842, 0.1340169
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1139517, 0.1139107
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0487736, 0.0487860
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0425491, 0.0425110

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 2175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056101, upper bound: 0.0056193
time: 6.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056106, upper bound: 0.0056181
time: 76.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1463260, 0.1463591
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170027, 0.0170021
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0875434, 0.0874428
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916561, 0.1916705
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1597900, 0.1597231
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989130, 0.1989311
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1339841, 0.1340171
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1139656, 0.1138968
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0487511, 0.0488086
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0425844, 0.0424757

Time for backsubstitution: 6.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056105, upper bound: 0.0056184
time: 51.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056112, upper bound: 0.0056183
time: 9.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1463486, 0.1463365
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170023, 0.0170024
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0874454, 0.0875407
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916644, 0.1916623
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1597265, 0.1597866
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989234, 0.1989207
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1340167, 0.1339844
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1139041, 0.1139583
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0487986, 0.0487610
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0424900, 0.0425701

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056172, upper bound: 0.0056117
time: 43.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056177, upper bound: 0.0056124
time: 4.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1463305, 0.1463546
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170027, 0.0170020
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0874505, 0.0875356
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916559, 0.1916708
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1597325, 0.1597806
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989135, 0.1989306
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1340166, 0.1339845
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1139181, 0.1139443
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0487761, 0.0487836
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0425253, 0.0425349

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056176, upper bound: 0.0056116
time: 8.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056183, upper bound: 0.0056103
time: 21.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1463546, 0.1463305
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170020, 0.0170027
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0875356, 0.0874505
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916708, 0.1916559
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1597806, 0.1597326
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989306, 0.1989135
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1339845, 0.1340166
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1139443, 0.1139181
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0487836, 0.0487761
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0425349, 0.0425253

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2175

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056101, upper bound: 0.0056190
time: 9.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056106, upper bound: 0.0056182
time: 16.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1463365, 0.1463486
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170024, 0.0170023
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0875407, 0.0874455
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916623, 0.1916644
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1597865, 0.1597265
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989207, 0.1989235
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1339844, 0.1340167
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1139583, 0.1139041
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0487610, 0.0487987
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0425701, 0.0424900

Time for backsubstitution: 6.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2175

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056105, upper bound: 0.0056186
time: 17.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056112, upper bound: 0.0056187
time: 5.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1463591, 0.1463261
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170021, 0.0170027
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0874428, 0.0875434
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916705, 0.1916562
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1597231, 0.1597900
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989311, 0.1989130
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1340170, 0.1339841
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1138968, 0.1139656
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0488086, 0.0487511
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0424757, 0.0425844

Time for backsubstitution: 6.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2175

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056172, upper bound: 0.0056118
time: 4.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056177, upper bound: 0.0056112
time: 5.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1463410, 0.1463441
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170025, 0.0170023
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0874479, 0.0875383
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916620, 0.1916646
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1597291, 0.1597840
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989211, 0.1989230
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1340169, 0.1339842
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1139107, 0.1139517
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0487860, 0.0487736
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0425110, 0.0425491

Time for backsubstitution: 6.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2175

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056176, upper bound: 0.0056110
time: 213.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056183, upper bound: 0.0056104
time: 21.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1463445, 0.1463407
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170023, 0.0170025
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0875385, 0.0874477
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916653, 0.1916614
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1597855, 0.1597276
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989240, 0.1989202
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1339844, 0.1340168
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1139532, 0.1139092
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0487733, 0.0487863
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0425496, 0.0425105

Time for backsubstitution: 7.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056105, upper bound: 0.0056191
time: 4.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056110, upper bound: 0.0056183
time: 6.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1463264, 0.1463588
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170027, 0.0170021
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0875436, 0.0874426
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916568, 0.1916699
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1597915, 0.1597216
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989140, 0.1989301
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1339842, 0.1340169
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1139671, 0.1138953
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0487508, 0.0488088
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0425849, 0.0424752

Time for backsubstitution: 7.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 2175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056109, upper bound: 0.0056182
time: 4.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056116, upper bound: 0.0056174
time: 40.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1463489, 0.1463362
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170023, 0.0170025
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0874456, 0.0875405
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916650, 0.1916617
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1597280, 0.1597851
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989244, 0.1989197
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1340169, 0.1339843
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1139056, 0.1139567
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0487984, 0.0487613
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0424905, 0.0425696

Time for backsubstitution: 7.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056176, upper bound: 0.0056104
time: 174.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056181, upper bound: 0.0056114
time: 3.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3333488, -0.8811949, -1.3333488, -0.8811949, -0.1463308, 0.1463543
1: 0.9464985, 1.0491189, 0.9464985, 1.0491189, -0.0170027, 0.0170020
2: -3.3222301, -2.8490586, -3.3222301, -2.8490586, -0.0874507, 0.0875354
3: -3.3652139, -2.6057148, -3.3652139, -2.6057148, -0.1916565, 0.1916701
4: -3.0320780, -2.4044356, -3.0320780, -2.4044356, -0.1597340, 0.1597791
5: -3.2069278, -2.4078224, -3.2069278, -2.4078224, -0.1989145, 0.1989297
6: -3.8119714, -3.1229112, -3.8119714, -3.1229112, -0.1340168, 0.1339844
7: -0.3961493, 0.2472235, -0.3961493, 0.2472235, -0.1139196, 0.1139428
8: -1.2221160, -0.9510399, -1.2221160, -0.9510399, -0.0487758, 0.0487838
9: -1.3623049, -1.1319052, -1.3623049, -1.1319052, -0.0425258, 0.0425344

Time for backsubstitution: 6.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 410
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2149
type: DSZ, layer: 1, pos: 588
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 619
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 149
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 753
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 1096
type: DSZ, layer: 1, pos: 1111
type: DSZ, layer: 1, pos: 1112
type: DSZ, layer: 1, pos: 1113
type: DSZ, layer: 1, pos: 1114
type: DSZ, layer: 1, pos: 1115
type: DSZ, layer: 1, pos: 1116
type: DSZ, layer: 1, pos: 1117
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2277
type: DSZ, layer: 1, pos: 2286
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2379
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2407
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2513
type: DSZ, layer: 1, pos: 2523
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2665
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2708
type: DSZ, layer: 1, pos: 2709
type: DSZ, layer: 1, pos: 2712
type: DSZ, layer: 1, pos: 2713
type: DSZ, layer: 1, pos: 2716
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2729
type: DSZ, layer: 1, pos: 2731
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2925
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2937
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2950
type: DSZ, layer: 1, pos: 2952
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2968
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3029
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3072
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3213
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3493

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056179, upper bound: 0.0056110
time: 169.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056187, upper bound: 0.0056104
time: 3.92 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 179.84 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056097, upper bound: 0.0056195
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056102, upper bound: 0.0056194
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056101, upper bound: 0.0056182
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056108, upper bound: 0.0056188
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056168, upper bound: 0.0056124
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056173, upper bound: 0.0056124
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056172, upper bound: 0.0056120
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056179, upper bound: 0.0056120
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056101, upper bound: 0.0056193
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056106, upper bound: 0.0056181
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056105, upper bound: 0.0056184
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056112, upper bound: 0.0056183
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056172, upper bound: 0.0056117
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056177, upper bound: 0.0056124
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056176, upper bound: 0.0056116
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056183, upper bound: 0.0056103
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056101, upper bound: 0.0056190
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056106, upper bound: 0.0056182
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056105, upper bound: 0.0056186
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056112, upper bound: 0.0056187
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056172, upper bound: 0.0056118
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056177, upper bound: 0.0056112
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056176, upper bound: 0.0056110
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056183, upper bound: 0.0056104
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056105, upper bound: 0.0056191
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056110, upper bound: 0.0056183
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056109, upper bound: 0.0056182
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056116, upper bound: 0.0056174
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056176, upper bound: 0.0056104
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056181, upper bound: 0.0056114
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056179, upper bound: 0.0056110
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 179.84
Output dim: 1, lower bound: -0.0056187, upper bound: 0.0056104

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 66.84 + 1867.80 = 1934.64 seconds
