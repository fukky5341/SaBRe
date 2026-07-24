## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 7)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.10315783889999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.5914955, -2.2146628, -3.5914955, -2.2146628, -0.5696697, 0.5696697)
1: (-4.1967015, -2.1738605, -4.1967015, -2.1738605, -0.8479316, 0.8479316)
2: (-0.7871203, -0.1651437, -0.7871203, -0.1651437, -0.5425514, 0.5425515)
3: (0.1599226, 0.6740252, 0.1599226, 0.6740252, -0.4912806, 0.4912806)
4: (-1.7076153, -0.7517804, -1.7076153, -0.7517804, -0.6395983, 0.6395983)
5: (-0.0337971, 0.5903980, -0.0337971, 0.5903980, -0.5007951, 0.5007952)
6: (-0.3379841, 0.5070201, -0.3379841, 0.5070201, -0.7854927, 0.7854928)
7: (-0.9320757, 0.4672965, -0.9320757, 0.4672965, -0.5549766, 0.5549766)
8: (-3.6381791, -1.4250402, -3.6381791, -1.4250402, -0.7067745, 0.7067746)
9: (-1.8199186, 0.0598726, -1.8199186, 0.0598726, -0.7099850, 0.7099849)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.29 + 175.84 = 184.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.1032555, upper bound: 0.1032611

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 377
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 412
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 3507
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 2784
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 3532
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2321
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2781
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2855
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 3261
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 2862
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2811
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 3487
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 3043
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2785
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2802
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 2768
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2208
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 3218
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2307
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2729
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3464

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2420

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1032063, upper bound: 0.1032486
time: 32.89 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1032383, upper bound: 0.1032510
time: 17.89 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 50.85 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 50.85
Output dim: 3, lower bound: -0.1032063, upper bound: 0.1032486
NS_A2, status: Status.UNKNOWN, split count: 1, time: 50.85
Output dim: 3, lower bound: -0.1032383, upper bound: 0.1032510

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.5914540, -2.2176163, -3.5914578, -2.2173219, -0.5670671, 0.5668851
1: -4.1966963, -2.1807094, -4.1966968, -2.1800666, -0.8417662, 0.8412179
2: -0.7862841, -0.1651633, -0.7863690, -0.1651614, -0.5414854, 0.5415749
3: 0.1610142, 0.6740202, 0.1609082, 0.6740206, -0.4900897, 0.4902011
4: -1.7056323, -0.7517946, -1.7058315, -0.7517934, -0.6374009, 0.6376112
5: -0.0325139, 0.5903971, -0.0326291, 0.5903971, -0.4995290, 0.4996571
6: -0.3366463, 0.5070171, -0.3367773, 0.5070173, -0.7841237, 0.7842579
7: -0.9302903, 0.4672911, -0.9304711, 0.4672916, -0.5534636, 0.5536064
8: -3.6381741, -1.4314742, -3.6381748, -1.4308419, -0.7014346, 0.7009571
9: -1.8199105, 0.0535707, -1.8199124, 0.0541644, -0.7041808, 0.7036525

Time for backsubstitution: 6.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 377
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3040
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 412
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 3507
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2839
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 263
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2490
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 2784
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2321
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2781
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2840
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2855
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 3261
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2811
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 3487
type: B, layer: 1, pos: 2810
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2785
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2802
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 2083
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 2768
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2208
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 3218
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2307
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2324
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2729
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3464

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2629

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1031514, upper bound: 0.1032166
time: 240.29 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1031947, upper bound: 0.1032334
time: 15.59 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.5963764, -2.2153919, -3.5914631, -2.2155809, -0.5719165, 0.5675594
1: -4.2075853, -2.1778388, -4.1966996, -2.1776400, -0.8545176, 0.8417231
2: -0.7864326, -0.1632400, -0.7864055, -0.1651530, -0.5415378, 0.5440301
3: 0.1601066, 0.6758401, 0.1605690, 0.6740212, -0.4908434, 0.4924377
4: -1.7068520, -0.7505244, -1.7067188, -0.7517871, -0.6387024, 0.6401375
5: -0.0334217, 0.5926936, -0.0332938, 0.5903972, -0.5000533, 0.5021751
6: -0.3376534, 0.5093730, -0.3372201, 0.5070186, -0.7851048, 0.7871317
7: -0.9315662, 0.4703425, -0.9315642, 0.4672941, -0.5536128, 0.5563554
8: -3.6489918, -1.4276562, -3.6381767, -1.4275215, -0.7123297, 0.7014259
9: -1.8295646, 0.0558348, -1.8199146, 0.0561643, -0.7157830, 0.7041156

Time for backsubstitution: 6.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 2396
type: B, layer: 1, pos: 377
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2644
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 3040
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2154
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 412
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2172
type: B, layer: 1, pos: 3507
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2839
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 2299
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 3112
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2636
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 263
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 3398
type: B, layer: 1, pos: 2490
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 2784
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2321
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2781
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2840
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2081
type: B, layer: 1, pos: 2855
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 3261
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 2862
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2811
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2233
type: B, layer: 1, pos: 3487
type: B, layer: 1, pos: 2810
type: B, layer: 1, pos: 2780
type: B, layer: 1, pos: 3043
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2785
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 2802
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 2083
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 2769
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 3186
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 2767
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 2768
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2208
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 3218
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 2307
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2234
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2324
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2369
type: B, layer: 1, pos: 2399
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2689
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2729
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3464

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2629

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1031933, upper bound: 0.1032212
time: 10.64 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1032231, upper bound: 0.1032279
time: 242.15 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 259.25 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 259.25
Output dim: 3, lower bound: -0.1031514, upper bound: 0.1032166
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 259.25
Output dim: 3, lower bound: -0.1031947, upper bound: 0.1032334
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 259.25
Output dim: 3, lower bound: -0.1031933, upper bound: 0.1032212
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 259.25
Output dim: 3, lower bound: -0.1032231, upper bound: 0.1032279

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3.5884471, -2.2178292, -3.5881367, -2.2184143, -0.5598018, 0.5620598
1: -4.1868515, -2.1807108, -4.1858501, -2.1832895, -0.8206887, 0.8268583
2: -0.7861959, -0.1673509, -0.7855042, -0.1675821, -0.5389661, 0.5384619
3: 0.1614564, 0.6723087, 0.1621201, 0.6721351, -0.4877734, 0.4872019
4: -1.7054582, -0.7533578, -1.7042732, -0.7535230, -0.6355491, 0.6345106
5: -0.0323817, 0.5886033, -0.0318477, 0.5884141, -0.4973691, 0.4968711
6: -0.3362268, 0.5049971, -0.3355025, 0.5048047, -0.7814678, 0.7809035
7: -0.9302664, 0.4656848, -0.9307943, 0.4655517, -0.5503907, 0.5497813
8: -3.6302156, -1.4314775, -3.6293819, -1.4336929, -0.6838562, 0.6890838
9: -1.8106382, 0.0535665, -1.8097243, 0.0502648, -0.6846913, 0.6905851

Time for backsubstitution: 6.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 377
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 412
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 3507
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 2784
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 3532
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2321
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2781
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2855
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 3261
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 2862
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2811
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 3487
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3043
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2785
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2802
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 2768
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2208
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 3218
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2307
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2729
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3464

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2396

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1031256, upper bound: 0.1031997
time: 285.70 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1031377, upper bound: 0.1032031
time: 130.11 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.5905528, -2.2176516, -3.5904658, -2.2173605, -0.5666981, 0.5595628
1: -4.1943588, -2.1807094, -4.1941233, -2.1800675, -0.8411701, 0.8194932
2: -0.7862667, -0.1654565, -0.7863498, -0.1654865, -0.5408266, 0.5412354
3: 0.1611041, 0.6736469, 0.1610072, 0.6736164, -0.4895599, 0.4898086
4: -1.7056009, -0.7520739, -1.7057967, -0.7520973, -0.6370974, 0.6373397
5: -0.0324857, 0.5899692, -0.0325979, 0.5899259, -0.4988171, 0.4992525
6: -0.3365744, 0.5065774, -0.3366972, 0.5065291, -0.7835704, 0.7837474
7: -0.9302793, 0.4664434, -0.9304594, 0.4663770, -0.5497673, 0.5534221
8: -3.6362150, -1.4314749, -3.6360173, -1.4308424, -0.7009558, 0.6827128
9: -1.8178267, 0.0535684, -1.8176153, 0.0541635, -0.7036144, 0.6833450

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 377
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 412
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 3507
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 2784
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 3532
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2321
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2781
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2855
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 3261
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 2862
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2811
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 3487
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 3043
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2785
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2802
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 2768
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2208
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 3218
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2307
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2729
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3464

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2396

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1031689, upper bound: 0.1032112
time: 16.86 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1031801, upper bound: 0.1032126
time: 247.48 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.5936031, -2.2155836, -3.5881457, -2.2159460, -0.5655549, 0.5627702
1: -4.1985598, -2.1778412, -4.1858525, -2.1788521, -0.8358855, 0.8273648
2: -0.7863533, -0.1652116, -0.7857354, -0.1675702, -0.5390019, 0.5413870
3: 0.1605048, 0.6742732, 0.1614360, 0.6721362, -0.4885476, 0.4899384
4: -1.7066908, -0.7519238, -1.7055802, -0.7535136, -0.6368607, 0.6376942
5: -0.0333021, 0.5910110, -0.0328849, 0.5884143, -0.4979007, 0.5000391
6: -0.3372751, 0.5075606, -0.3364147, 0.5048069, -0.7824850, 0.7844657
7: -0.9315439, 0.4688452, -0.9322593, 0.4655550, -0.5505418, 0.5529631
8: -3.6417842, -1.4276588, -3.6293845, -1.4286718, -0.6967877, 0.6895545
9: -1.8211062, 0.0558319, -1.8097291, 0.0540624, -0.6984943, 0.6910505

Time for backsubstitution: 6.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 377
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 412
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 3507
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 2784
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 3532
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2321
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2781
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2855
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 3261
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 2862
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2811
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 3487
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 3043
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2785
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2802
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 2768
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2208
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 3218
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2307
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2729
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3464

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2396

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1031668, upper bound: 0.1032017
time: 100.16 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1031776, upper bound: 0.1032042
time: 155.94 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.5951562, -2.2154441, -3.5904710, -2.2156200, -0.5712748, 0.5601897
1: -4.2044582, -2.1778402, -4.1941242, -2.1776395, -0.8534049, 0.8199554
2: -0.7864082, -0.1636995, -0.7863864, -0.1654783, -0.5408804, 0.5434650
3: 0.1602371, 0.6753309, 0.1606682, 0.6736169, -0.4902888, 0.4918816
4: -1.7068081, -0.7508942, -1.7066841, -0.7520914, -0.6383940, 0.6397420
5: -0.0333814, 0.5921236, -0.0332625, 0.5899260, -0.4993450, 0.5015922
6: -0.3375475, 0.5087351, -0.3371403, 0.5065304, -0.7845201, 0.7864228
7: -0.9315545, 0.4693718, -0.9315523, 0.4663793, -0.5499111, 0.5560489
8: -3.6463773, -1.4276571, -3.6360192, -1.4275210, -0.7114102, 0.6831343
9: -1.8267574, 0.0558338, -1.8176186, 0.0561633, -0.7147787, 0.6837659

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2396
type: A, layer: 1, pos: 377
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 2644
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 3040
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2154
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 412
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2172
type: A, layer: 1, pos: 3507
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2839
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 2299
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 3112
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2636
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 3398
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 338
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 2784
type: A, layer: 1, pos: 3532
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 2321
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2781
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 2840
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2081
type: A, layer: 1, pos: 2855
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 3261
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2862
type: A, layer: 1, pos: 2811
type: A, layer: 1, pos: 2233
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 3487
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 2780
type: A, layer: 1, pos: 3043
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2785
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 2802
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 2083
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 2769
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 3186
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 2767
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 2768
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2208
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 3218
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 2307
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2234
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2324
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2369
type: A, layer: 1, pos: 2399
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2689
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2729
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3464

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2396

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1031968, upper bound: 0.1032091
time: 143.20 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.1032083, upper bound: 0.1032160
time: 20.08 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 169.36 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 169.36
Output dim: 3, lower bound: -0.1031256, upper bound: 0.1031997
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 169.36
Output dim: 3, lower bound: -0.1031377, upper bound: 0.1032031
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 169.36
Output dim: 3, lower bound: -0.1031689, upper bound: 0.1032112
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 169.36
Output dim: 3, lower bound: -0.1031801, upper bound: 0.1032126
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 169.36
Output dim: 3, lower bound: -0.1031668, upper bound: 0.1032017
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 169.36
Output dim: 3, lower bound: -0.1031776, upper bound: 0.1032042
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 169.36
Output dim: 3, lower bound: -0.1031968, upper bound: 0.1032091
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 169.36
Output dim: 3, lower bound: -0.1032083, upper bound: 0.1032160

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 184.13 + 1696.74 = 1880.87 seconds
