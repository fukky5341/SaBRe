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
execution time: IAR + RelationalAnalysis = 7.16 + 57.37 = 64.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0056239, upper bound: 0.0056236

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 3213
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2723
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 380
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2708
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2717
type: A, layer: 1, pos: 2703
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2712
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2716
type: A, layer: 1, pos: 2709
type: A, layer: 1, pos: 2742
type: A, layer: 1, pos: 2743
type: A, layer: 1, pos: 2938
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 2713
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2731
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 2277
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 2415
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 2952
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 1113
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2729
type: A, layer: 1, pos: 2744
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3269

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2346

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056183, upper bound: 0.0055985
time: 5.54 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056179, upper bound: 0.0056194
time: 3.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.93 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.93
Output dim: 1, lower bound: -0.0056183, upper bound: 0.0055985
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.93
Output dim: 1, lower bound: -0.0056179, upper bound: 0.0056194

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1.3330665, -0.8823897, -1.3333424, -0.8822398, -0.1450491, 0.1456410
1: 0.9466289, 1.0487251, 0.9464985, 1.0487757, -0.0163181, 0.0165111
2: -3.3215876, -2.8491251, -3.3216829, -2.8490589, -0.0886966, 0.0885038
3: -3.3619843, -2.6063747, -3.3624058, -2.6057167, -0.1870643, 0.1851685
4: -3.0313787, -2.4044166, -3.0314841, -2.4044356, -0.1601461, 0.1597850
5: -3.2035031, -2.4085298, -3.2039471, -2.4078236, -0.1938529, 0.1917931
6: -3.8090370, -3.1238737, -3.8093863, -3.1229112, -0.1304351, 0.1286582
7: -0.3948807, 0.2473340, -0.3950195, 0.2472235, -0.1134189, 0.1125136
8: -1.2221997, -0.9512641, -1.2221162, -0.9512414, -0.0496444, 0.0497861
9: -1.3621945, -1.1319293, -1.3622100, -1.1319063, -0.0450178, 0.0449552

Time for backsubstitution: 5.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 3213
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 2723
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 380
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2708
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2717
type: B, layer: 1, pos: 2703
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2712
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 2709
type: B, layer: 1, pos: 2742
type: B, layer: 1, pos: 2743
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 2713
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 2968
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 2277
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2415
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 2952
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 1113
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2729
type: B, layer: 1, pos: 2744
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3269

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3039

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056078, upper bound: 0.0055751
time: 8.74 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0055947, upper bound: 0.0055749
time: 3.18 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1.3333484, -0.8813363, -1.3333484, -0.8813100, -0.1472870, 0.1448579
1: 0.9464985, 1.0490906, 0.9464985, 1.0490961, -0.0170144, 0.0162736
2: -3.3220863, -2.8490586, -3.3221116, -2.8490586, -0.0884169, 0.0894625
3: -3.3648124, -2.6057150, -3.3648915, -2.6057146, -0.1846853, 0.1918699
4: -3.0318973, -2.4044356, -3.0319290, -2.4044356, -0.1597089, 0.1611811
5: -3.2065029, -2.4078224, -3.2065866, -2.4078224, -0.1912473, 0.1990304
6: -3.8117545, -3.1229112, -3.8117967, -3.1229112, -0.1281821, 0.1346404
7: -0.3958017, 0.2472235, -0.3958691, 0.2472235, -0.1122161, 0.1157338
8: -1.2221161, -0.9511361, -1.2221162, -0.9511198, -0.0502250, 0.0495841
9: -1.3622932, -1.1319056, -1.3622952, -1.1319054, -0.0449342, 0.0451884

Time for backsubstitution: 5.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 410
type: B, layer: 1, pos: 2379
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2407
type: B, layer: 1, pos: 2286
type: B, layer: 1, pos: 2149
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2673
type: B, layer: 1, pos: 3213
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 2365
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 2723
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 380
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 3072
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2708
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2523
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2717
type: B, layer: 1, pos: 2703
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2712
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2716
type: B, layer: 1, pos: 2709
type: B, layer: 1, pos: 2742
type: B, layer: 1, pos: 2743
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2937
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 2713
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 2968
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2731
type: B, layer: 1, pos: 2950
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 2509
type: B, layer: 1, pos: 2955
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 2277
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2415
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 2952
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 2925
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 1096
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 149
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1111
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 1113
type: B, layer: 1, pos: 1114
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1117
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2729
type: B, layer: 1, pos: 2744
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3029
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3269

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3039

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0056080, upper bound: 0.0055956
time: 22.48 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0055951, upper bound: 0.0055956
time: 9.15 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 37.21 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 37.21
Output dim: 1, lower bound: -0.0056078, upper bound: 0.0055751
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 37.21
Output dim: 1, lower bound: -0.0055947, upper bound: 0.0055749
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 37.21
Output dim: 1, lower bound: -0.0056080, upper bound: 0.0055956
NS_A2_B2, status: Status.VERIFIED, split count: 2, time: 37.21
Output dim: 1, lower bound: -0.0055951, upper bound: 0.0055956

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -1.3330635, -0.8826240, -1.3352164, -0.8825105, -0.1424493, 0.1420398
1: 0.9466289, 1.0486107, 0.9462578, 1.0486392, -0.0157807, 0.0157621
2: -3.3215213, -2.8491251, -3.3216062, -2.8481250, -0.0871304, 0.0873718
3: -3.3615079, -2.6063750, -3.3618665, -2.6005330, -0.1783443, 0.1788591
4: -3.0313144, -2.4044166, -3.0314045, -2.4030962, -0.1580940, 0.1582956
5: -3.2029297, -2.4085302, -3.2033076, -2.4026458, -0.1845290, 0.1850461
6: -3.8071542, -3.1238737, -3.8071549, -3.1209624, -0.1226206, 0.1229824
7: -0.3946832, 0.2473341, -0.3947845, 0.2496288, -0.1094787, 0.1096650
8: -1.2221997, -0.9513064, -1.2228625, -0.9512936, -0.0488416, 0.0486780
9: -1.3621764, -1.1319298, -1.3621876, -1.1317447, -0.0447349, 0.0447523

Time for backsubstitution: 5.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 3213
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2723
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 380
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2708
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2717
type: A, layer: 1, pos: 2703
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2712
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2716
type: A, layer: 1, pos: 2709
type: A, layer: 1, pos: 2742
type: A, layer: 1, pos: 2743
type: A, layer: 1, pos: 2938
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 2713
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2731
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 2277
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 2415
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 2952
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 1113
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2729
type: A, layer: 1, pos: 2744
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3269

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2544

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0055945, upper bound: 0.0055633
time: 70.17 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0055963, upper bound: 0.0055639
time: 4.34 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1.3333449, -0.8815707, -1.3352219, -0.8815807, -0.1446870, 0.1412567
1: 0.9464985, 1.0489763, 0.9462578, 1.0489597, -0.0164770, 0.0155246
2: -3.3220198, -2.8490586, -3.3220341, -2.8481250, -0.0868507, 0.0883306
3: -3.3643360, -2.6057158, -3.3643522, -2.6005330, -0.1759653, 0.1855604
4: -3.0318322, -2.4044356, -3.0318489, -2.4030962, -0.1576567, 0.1596915
5: -3.2059295, -2.4078236, -3.2059474, -2.4026446, -0.1819234, 0.1922832
6: -3.8098712, -3.1229112, -3.8095646, -3.1209624, -0.1203676, 0.1289645
7: -0.3956047, 0.2472235, -0.3956336, 0.2496288, -0.1082759, 0.1128853
8: -1.2221160, -0.9511783, -1.2228625, -0.9511718, -0.0494222, 0.0484760
9: -1.3622752, -1.1319058, -1.3622730, -1.1317437, -0.0446512, 0.0449854

Time for backsubstitution: 5.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 410
type: A, layer: 1, pos: 2379
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2407
type: A, layer: 1, pos: 2286
type: A, layer: 1, pos: 2149
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 2673
type: A, layer: 1, pos: 3213
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 2365
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 2723
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 380
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 3072
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2708
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2523
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2717
type: A, layer: 1, pos: 2703
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2712
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2716
type: A, layer: 1, pos: 2709
type: A, layer: 1, pos: 2742
type: A, layer: 1, pos: 2743
type: A, layer: 1, pos: 2938
type: A, layer: 1, pos: 2937
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 2713
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2968
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2731
type: A, layer: 1, pos: 2950
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 2509
type: A, layer: 1, pos: 2955
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 2277
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 2415
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 2952
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 2925
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 1096
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 149
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1111
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 1113
type: A, layer: 1, pos: 1114
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1117
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2729
type: A, layer: 1, pos: 2744
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3029
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3269

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2544

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0055945, upper bound: 0.0055844
time: 20.59 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0055965, upper bound: 0.0055844
time: 5.22 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.39 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 31.39
Output dim: 1, lower bound: -0.0055945, upper bound: 0.0055633
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 31.39
Output dim: 1, lower bound: -0.0055963, upper bound: 0.0055639
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 31.39
Output dim: 1, lower bound: -0.0055945, upper bound: 0.0055844
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 31.39
Output dim: 1, lower bound: -0.0055965, upper bound: 0.0055844

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 64.53 + 175.12 = 239.65 seconds
