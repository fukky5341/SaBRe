## Execution arguments:
Dataset: Dataset.CIFAR10
Network: onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 5)
Time budget: 7200 seconds
Split limit: 100
Threshold: 0.12740707405


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.2255214, 0.3759673, -0.2255214, 0.3759673, -0.2810238, 0.2810238)
1: (-1.1485041, -0.1843055, -1.1485041, -0.1843055, -0.4201857, 0.4201857)
2: (-1.8206991, -0.5255621, -1.8206991, -0.5255621, -0.9402038, 0.9402038)
3: (-3.4475076, -1.1946113, -3.4475076, -1.1946113, -1.3280172, 1.3280175)
4: (-3.2462273, -1.3580585, -3.2462273, -1.3580585, -0.6059348, 0.6059347)
5: (-4.1147146, -1.7567466, -4.1147146, -1.7567466, -1.4206862, 1.4206861)
6: (-5.6424732, -2.8801653, -5.6424732, -2.8801653, -1.7451348, 1.7451348)
7: (-6.1008816, -3.9028692, -6.1008816, -3.9028692, -0.8280121, 0.8280121)
8: (0.3400624, 1.0824537, 0.3400624, 1.0824537, -0.6312646, 0.6312646)
9: (-1.0499132, 0.0607006, -1.0499132, 0.0607006, -0.8351963, 0.8351963)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 5.78 + 458.01 = 463.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1276895, upper bound: 0.1276922

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 393
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 2574
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 2122
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3512
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 3572
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 3574
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2683
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 3043
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3154
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 3366

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 3470

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1275057, upper bound: 0.1276903
time: 193.63 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276875, upper bound: 0.1276916
time: 155.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 349.47 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 349.47
Output dim: 8, lower bound: -0.1275057, upper bound: 0.1276903
IS_A2, status: Status.UNKNOWN, split count: 1, time: 349.47
Output dim: 8, lower bound: -0.1276875, upper bound: 0.1276916

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.2249140, 0.3746682, -0.2255015, 0.3749962, -0.2791663, 0.2796432
1: -1.1480010, -0.1849663, -1.1483967, -0.1846964, -0.4192517, 0.4194272
2: -1.8174086, -0.5312263, -1.8206484, -0.5299727, -0.9320277, 0.9342331
3: -3.4487464, -1.1955892, -3.4474359, -1.1949987, -1.3282251, 1.3268712
4: -3.2454174, -1.3602344, -3.2459493, -1.3596159, -0.6023623, 0.6029788
5: -4.1163754, -1.7577686, -4.1146235, -1.7571216, -1.4206488, 1.4194385
6: -5.6350374, -2.8834858, -5.6377316, -2.8801877, -1.7378812, 1.7352136
7: -6.1045737, -3.9046905, -6.1008363, -3.9035501, -0.8239858, 0.8239210
8: 0.3445393, 1.0799770, 0.3433519, 1.0824529, -0.6269473, 0.6254684
9: -1.0490155, 0.0600339, -1.0492862, 0.0605691, -0.8341527, 0.8338634

Time for backsubstitution: 4.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 393
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 2574
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3512
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 3572
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2683
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3154
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3366

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 393

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1275063, upper bound: 0.1274896
time: 653.21 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1274981, upper bound: 0.1276894
time: 201.54 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.2255173, 0.3759464, -0.2255182, 0.3759489, -0.2810167, 0.2810164
1: -1.1485033, -0.1843411, -1.1485038, -0.1843344, -0.4201586, 0.4201461
2: -1.8206981, -0.5256109, -1.8206983, -0.5256033, -0.9401746, 0.9401358
3: -3.4475029, -1.1955553, -3.4475040, -1.1953753, -1.3273301, 1.3272274
4: -3.2462225, -1.3583803, -3.2462237, -1.3583394, -0.6057168, 0.6056535
5: -4.1147099, -1.7578514, -4.1147113, -1.7576407, -1.4199500, 1.4198105
6: -5.6420193, -2.8801658, -5.6421056, -2.8801644, -1.7450299, 1.7450645
7: -6.1008816, -3.9054041, -6.1008811, -3.9049203, -0.8273696, 0.8271798
8: 0.3400725, 1.0824538, 0.3400708, 1.0824538, -0.6312485, 0.6312534
9: -1.0499077, 0.0606997, -1.0499086, 0.0606998, -0.8351924, 0.8351946

Time for backsubstitution: 4.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 393
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 2574
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3512
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 3572
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2683
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3154
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3366

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 393

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276878, upper bound: 0.1274941
time: 38.88 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276880, upper bound: 0.1276974
time: 22.01 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 65.12 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 65.12
Output dim: 8, lower bound: -0.1275063, upper bound: 0.1274896
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 65.12
Output dim: 8, lower bound: -0.1274981, upper bound: 0.1276894
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 65.12
Output dim: 8, lower bound: -0.1276878, upper bound: 0.1274941
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 65.12
Output dim: 8, lower bound: -0.1276880, upper bound: 0.1276974

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.2248397, 0.3679272, -0.2218093, 0.3664594, -0.2708617, 0.2695160
1: -1.1399459, -0.1849694, -1.1379980, -0.1889265, -0.4067173, 0.4088692
2: -1.8173268, -0.5393881, -1.8162112, -0.5403585, -0.9216462, 0.9216911
3: -3.4480715, -1.1956369, -3.4466250, -1.1952221, -1.3271165, 1.3260152
4: -3.2452214, -1.3627956, -3.2465472, -1.3624066, -0.5975875, 0.5978136
5: -4.1162796, -1.7578112, -4.1145287, -1.7572287, -1.4202921, 1.4192761
6: -5.6341724, -2.8835177, -5.6367331, -2.8804185, -1.7363453, 1.7340834
7: -6.1044827, -3.9082439, -6.1025510, -3.9072862, -0.8191614, 0.8197904
8: 0.3462281, 1.0799754, 0.3455611, 1.0817221, -0.6244838, 0.6232408
9: -1.0400476, 0.0599665, -1.0376098, 0.0557093, -0.8203237, 0.8221418

Time for backsubstitution: 4.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 2574
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 393
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 2122
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3512
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 3572
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 3574
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2683
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 3043
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3154
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 3366

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1274403, upper bound: 0.1274875
time: 319.17 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1275071, upper bound: 0.1274913
time: 104.79 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.2249118, 0.3746586, -0.2254989, 0.3749835, -0.2685859, 0.2796192
1: -1.1479934, -0.1849663, -1.1483870, -0.1846970, -0.4192470, 0.4131223
2: -1.8173997, -0.5312474, -1.8206369, -0.5299999, -0.9255105, 0.9341983
3: -3.4487205, -1.1955917, -3.4474030, -1.1950015, -1.3281007, 1.3257102
4: -3.2454095, -1.3605629, -3.2459407, -1.3599913, -0.5972549, 0.6027340
5: -4.1163712, -1.7577707, -4.1146173, -1.7571243, -1.4205861, 1.4191588
6: -5.6349669, -2.8834865, -5.6376400, -2.8801885, -1.7376089, 1.7334988
7: -6.1045728, -3.9052014, -6.1008348, -3.9042146, -0.8207301, 0.8233975
8: 0.3445491, 1.0799770, 0.3433644, 1.0824528, -0.6269348, 0.6240702
9: -1.0490105, 0.0600305, -1.0492802, 0.0605662, -0.8341452, 0.8320411

Time for backsubstitution: 4.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 2574
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 393
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 2122
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3512
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 3572
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 3574
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2683
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 3043
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 3154
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 3366

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1274439, upper bound: 0.1276882
time: 80.20 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1275002, upper bound: 0.1276897
time: 29.04 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.2254433, 0.3692052, -0.2218263, 0.3674121, -0.2727120, 0.2708893
1: -1.1404505, -0.1843442, -1.1381053, -0.1885645, -0.4076264, 0.4095889
2: -1.8206166, -0.5337728, -1.8162622, -0.5359904, -0.9297926, 0.9275934
3: -3.4468279, -1.1956018, -3.4466920, -1.1956006, -1.3262212, 1.3263710
4: -3.2460282, -1.3609418, -3.2468228, -1.3611304, -0.6009421, 0.6004897
5: -4.1146145, -1.7578944, -4.1146173, -1.7577469, -1.4195933, 1.4196483
6: -5.6411529, -2.8801970, -5.6411076, -2.8803971, -1.7434947, 1.7439346
7: -6.1007910, -3.9089575, -6.1025977, -3.9086561, -0.8225452, 0.8230502
8: 0.3417616, 1.0824521, 0.3422803, 1.0817230, -0.6287854, 0.6290257
9: -1.0409391, 0.0606321, -1.0382295, 0.0558400, -0.8213627, 0.8234708

Time for backsubstitution: 4.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 2574
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 393
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 2122
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3512
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 3572
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 3574
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2683
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 3043
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3154
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 3366

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276217, upper bound: 0.1274857
time: 273.08 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276806, upper bound: 0.1274908
time: 254.58 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.2255152, 0.3759368, -0.2255157, 0.3759366, -0.2704363, 0.2809923
1: -1.1484962, -0.1843413, -1.1484940, -0.1843344, -0.4201543, 0.4138410
2: -1.8206896, -0.5256320, -1.8206875, -0.5256305, -0.9336578, 0.9401013
3: -3.4474769, -1.1955576, -3.4474697, -1.1953781, -1.3272049, 1.3260660
4: -3.2462156, -1.3587089, -3.2462146, -1.3587151, -0.6006092, 0.6054086
5: -4.1147046, -1.7578533, -4.1147046, -1.7576433, -1.4198871, 1.4195307
6: -5.6419487, -2.8801677, -5.6420140, -2.8801663, -1.7447581, 1.7433491
7: -6.1008811, -3.9059155, -6.1008806, -3.9055848, -0.8241135, 0.8266568
8: 0.3400822, 1.0824535, 0.3400835, 1.0824537, -0.6312361, 0.6298549
9: -1.0499027, 0.0606969, -1.0499022, 0.0606965, -0.8351845, 0.8333720

Time for backsubstitution: 4.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 2574
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3248
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 393
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 2605
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 2122
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3512
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2660
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2291
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 3273
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 3151
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2640
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 3572
type: A, layer: 1, pos: 3152
type: A, layer: 1, pos: 3574
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2683
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 3103
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2143
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 3043
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3031
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2553
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 3150
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 2120
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3154
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2245
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2549
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 3366

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 153

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276263, upper bound: 0.1276869
time: 230.91 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276859, upper bound: 0.1274931
time: 382.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 617.77 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 617.77
Output dim: 8, lower bound: -0.1274403, upper bound: 0.1274875
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 617.77
Output dim: 8, lower bound: -0.1275071, upper bound: 0.1274913
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 617.77
Output dim: 8, lower bound: -0.1274439, upper bound: 0.1276882
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 617.77
Output dim: 8, lower bound: -0.1275002, upper bound: 0.1276897
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 617.77
Output dim: 8, lower bound: -0.1276217, upper bound: 0.1274857
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 617.77
Output dim: 8, lower bound: -0.1276806, upper bound: 0.1274908
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 617.77
Output dim: 8, lower bound: -0.1276263, upper bound: 0.1276869
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 617.77
Output dim: 8, lower bound: -0.1276859, upper bound: 0.1274931

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2248023, 0.3679173, -0.2217817, 0.3664518, -0.2708064, 0.2694525
1: -1.1396558, -0.1849282, -1.1377618, -0.1889304, -0.4063587, 0.4085102
2: -1.8166944, -0.5394635, -1.8157173, -0.5404863, -0.9207614, 0.9210490
3: -3.4448857, -1.1964211, -3.4440625, -1.1952300, -1.3239074, 1.3226023
4: -3.2439549, -1.3628918, -3.2455273, -1.3624071, -0.5963156, 0.5966098
5: -4.1127892, -1.7587309, -4.1117363, -1.7572329, -1.4167895, 1.4154981
6: -5.6318259, -2.8841710, -5.6348457, -2.8804259, -1.7339809, 1.7315114
7: -6.1026483, -3.9073205, -6.1010818, -3.9072869, -0.8162273, 0.8166530
8: 0.3471532, 1.0795267, 0.3463113, 1.0817136, -0.6235788, 0.6221904
9: -1.0400494, 0.0599276, -1.0376066, 0.0556847, -0.8201365, 0.8220582

Time for backsubstitution: 4.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 2574
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3512
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 3572
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2683
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3154
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3366

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 3470

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1274462, upper bound: 0.1273369
time: 19.13 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1274419, upper bound: 0.1274922
time: 100.33 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2248337, 0.3679248, -0.2218046, 0.3664576, -0.2708403, 0.2695163
1: -1.1398065, -0.1849704, -1.1378866, -0.1889272, -0.4063455, 0.4087697
2: -1.8172083, -0.5394198, -1.8161180, -0.5403839, -0.9215137, 0.9214913
3: -3.4474428, -1.1956378, -3.4461234, -1.1952244, -1.3249657, 1.3255236
4: -3.2449682, -1.3627954, -3.2463455, -1.3624067, -0.5966628, 0.5976057
5: -4.1155920, -1.7578119, -4.1139803, -1.7572297, -1.4178677, 1.4187416
6: -5.6337142, -2.8835194, -5.6363688, -2.8804207, -1.7356186, 1.7337219
7: -6.1032228, -3.9082439, -6.1015472, -3.9072862, -0.8157235, 0.8193067
8: 0.3462699, 1.0799736, 0.3455968, 1.0817206, -0.6240563, 0.6232030
9: -1.0400459, 0.0599596, -1.0376089, 0.0557041, -0.8203295, 0.8220276

Time for backsubstitution: 4.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 2574
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3512
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 3572
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2683
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3154
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3366

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 3470

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1275019, upper bound: 0.1273285
time: 462.72 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1275043, upper bound: 0.1274892
time: 400.96 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2248744, 0.3746484, -0.2254711, 0.3749756, -0.2685302, 0.2795554
1: -1.1477028, -0.1849253, -1.1481509, -0.1847005, -0.4188944, 0.4127609
2: -1.8167666, -0.5313259, -1.8201404, -0.5301275, -0.9246247, 0.9335482
3: -3.4455371, -1.1963778, -3.4448416, -1.1950076, -1.3248920, 1.3222986
4: -3.2441430, -1.3606584, -3.2449212, -1.3599918, -0.5959836, 0.6015288
5: -4.1128798, -1.7586882, -4.1118231, -1.7571280, -1.4170823, 1.4153805
6: -5.6326208, -2.8841405, -5.6357508, -2.8801956, -1.7352445, 1.7309268
7: -6.1027384, -3.9042768, -6.0993667, -3.9042153, -0.8177916, 0.8202637
8: 0.3454728, 1.0795281, 0.3441145, 1.0824437, -0.6260314, 0.6230182
9: -1.0489519, 0.0599922, -1.0492196, 0.0605415, -0.8339111, 0.8319131

Time for backsubstitution: 4.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 2574
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3512
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 3572
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2683
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3154
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3366

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 3470

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1274418, upper bound: 0.1275306
time: 281.92 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1274468, upper bound: 0.1276909
time: 515.48 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2249058, 0.3746558, -0.2254941, 0.3749813, -0.2685642, 0.2796192
1: -1.1478535, -0.1849672, -1.1482754, -0.1846977, -0.4188929, 0.4130223
2: -1.8172814, -0.5312793, -1.8205426, -0.5300254, -0.9253805, 0.9339984
3: -3.4480915, -1.1955931, -3.4469013, -1.1950021, -1.3259499, 1.3252184
4: -3.2451558, -1.3605629, -3.2457376, -1.3599917, -0.5963309, 0.6025254
5: -4.1156816, -1.7577713, -4.1140680, -1.7571255, -1.4181612, 1.4186238
6: -5.6345086, -2.8834884, -5.6372747, -2.8801904, -1.7368822, 1.7331374
7: -6.1033120, -3.9052017, -6.0998311, -3.9042153, -0.8172885, 0.8229118
8: 0.3445911, 1.0799749, 0.3434001, 1.0824509, -0.6265088, 0.6240324
9: -1.0489483, 0.0600240, -1.0492208, 0.0605606, -0.8341045, 0.8318830

Time for backsubstitution: 4.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 2574
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3512
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 3572
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2683
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3154
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3366

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 3470

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1275049, upper bound: 0.1275277
time: 363.66 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1275068, upper bound: 0.1276886
time: 888.47 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2254058, 0.3691954, -0.2217985, 0.3674043, -0.2726566, 0.2708259
1: -1.1401607, -0.1843035, -1.1378696, -0.1885680, -0.4072677, 0.4092298
2: -1.8199835, -0.5338482, -1.8157678, -0.5361186, -0.9289070, 0.9269509
3: -3.4436431, -1.1963865, -3.4441290, -1.1956071, -1.3230128, 1.3229582
4: -3.2447588, -1.3610351, -3.2458026, -1.3611302, -0.5996696, 0.5992859
5: -4.1111231, -1.7588124, -4.1118231, -1.7577506, -1.4160903, 1.4158703
6: -5.6388044, -2.8808501, -5.6392193, -2.8804038, -1.7411292, 1.7413632
7: -6.0989561, -3.9080329, -6.1011291, -3.9086561, -0.8196113, 0.8199126
8: 0.3426867, 1.0820034, 0.3430309, 1.0817147, -0.6278800, 0.6279749
9: -1.0409411, 0.0605941, -1.0382262, 0.0558158, -0.8211761, 0.8233874

Time for backsubstitution: 4.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 2574
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3512
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 3572
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2683
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3154
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3366

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 3470

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276224, upper bound: 0.1273082
time: 159.22 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276173, upper bound: 0.1273123
time: 500.82 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2254373, 0.3692029, -0.2218215, 0.3674101, -0.2726906, 0.2708896
1: -1.1403109, -0.1843449, -1.1379942, -0.1885652, -0.4072548, 0.4094894
2: -1.8204983, -0.5338043, -1.8161688, -0.5360152, -0.9296604, 0.9273942
3: -3.4461987, -1.1956030, -3.4461911, -1.1956012, -1.3240712, 1.3258790
4: -3.2457745, -1.3609421, -3.2466209, -1.3611302, -0.6000173, 0.6002820
5: -4.1139264, -1.7578954, -4.1140685, -1.7577479, -1.4171686, 1.4191134
6: -5.6406975, -2.8801987, -5.6407413, -2.8803968, -1.7427676, 1.7435730
7: -6.0995307, -3.9089575, -6.1015940, -3.9086561, -0.8191074, 0.8225663
8: 0.3418033, 1.0824498, 0.3423157, 1.0817214, -0.6283576, 0.6289883
9: -1.0409377, 0.0606257, -1.0382278, 0.0558353, -0.8213688, 0.8233570

Time for backsubstitution: 4.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 2574
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3248
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 2605
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 2122
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3512
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2660
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2291
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 3273
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2625
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 3151
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2640
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 3572
type: B, layer: 1, pos: 3152
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2683
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 3103
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2143
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 3031
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2553
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 3150
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2120
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3154
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2245
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2549
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 3366

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 3470

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276811, upper bound: 0.1273157
time: 35.20 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1276855, upper bound: 0.1273040
time: 32.38 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 71.81 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 71.81
Output dim: 8, lower bound: -0.1274462, upper bound: 0.1273369
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 71.81
Output dim: 8, lower bound: -0.1274419, upper bound: 0.1274922
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 71.81
Output dim: 8, lower bound: -0.1275019, upper bound: 0.1273285
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 71.81
Output dim: 8, lower bound: -0.1275043, upper bound: 0.1274892
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 71.81
Output dim: 8, lower bound: -0.1274418, upper bound: 0.1275306
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 71.81
Output dim: 8, lower bound: -0.1274468, upper bound: 0.1276909
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 71.81
Output dim: 8, lower bound: -0.1275049, upper bound: 0.1275277
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 71.81
Output dim: 8, lower bound: -0.1275068, upper bound: 0.1276886
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 71.81
Output dim: 8, lower bound: -0.1276224, upper bound: 0.1273082
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 71.81
Output dim: 8, lower bound: -0.1276173, upper bound: 0.1273123
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 71.81
Output dim: 8, lower bound: -0.1276811, upper bound: 0.1273157
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 71.81
Output dim: 8, lower bound: -0.1276855, upper bound: 0.1273040
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 71.81
Output dim: 8, lower bound: -0.1276263, upper bound: 0.1276869
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 71.81
Output dim: 8, lower bound: -0.1276859, upper bound: 0.1274931

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 463.78 + 6751.12 = 7214.90 seconds
