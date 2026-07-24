## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0250083666


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.6959207, -2.2369895, -3.6959207, -2.2369895, -0.6435004, 0.6435004)
1: (-3.8942285, -2.2368288, -3.8942285, -2.2368288, -0.6984701, 0.6984701)
2: (-2.2234130, -1.6189097, -2.2234130, -1.6189097, -0.4837343, 0.4837343)
3: (0.4531021, 0.7521143, 0.4531021, 0.7521143, -0.1814818, 0.1814818)
4: (-3.0147476, -2.2268522, -3.0147476, -2.2268522, -0.3151595, 0.3151595)
5: (0.8824778, 1.2126909, 0.8824778, 1.2126909, -0.1497966, 0.1497966)
6: (-2.1779187, -1.4503413, -2.1779187, -1.4503413, -0.1960290, 0.1960290)
7: (-1.8795151, -0.9322217, -1.8795151, -0.9322217, -0.3952203, 0.3952203)
8: (-3.9263482, -2.1335392, -3.9263482, -2.1335392, -0.7562028, 0.7562029)
9: (-4.5595870, -3.0845213, -4.5595870, -3.0845213, -0.5464629, 0.5464628)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.84 + 35.38 = 43.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0250319, upper bound: 0.0250374

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 288
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3454
type: A, layer: 1, pos: 3085
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 303
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 2109
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 3534
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3268
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 2068
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 2308
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 3012
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3177
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2256
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 2343
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2487
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 3163
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2309
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 3239
type: A, layer: 1, pos: 3524

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 341

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0249247, upper bound: 0.0250314
time: 14.39 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250252, upper bound: 0.0250237
time: 438.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 453.12 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 453.12
Output dim: 3, lower bound: -0.0249247, upper bound: 0.0250314
NS_A2, status: Status.UNKNOWN, split count: 1, time: 453.12
Output dim: 3, lower bound: -0.0250252, upper bound: 0.0250237

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.6958795, -2.2372754, -3.6958835, -2.2372446, -0.6432258, 0.6431998
1: -3.8942282, -2.2385497, -3.8942282, -2.2383683, -0.6967611, 0.6965603
2: -2.2205446, -1.6189163, -2.2208474, -1.6189154, -0.4811915, 0.4814249
3: 0.4551718, 0.7521114, 0.4549534, 0.7521117, -0.1793948, 0.1796145
4: -3.0141335, -2.2268522, -3.0141881, -2.2268522, -0.3144834, 0.3145479
5: 0.8840557, 1.2126880, 0.8838890, 1.2126883, -0.1481988, 0.1483671
6: -2.1743178, -1.4504056, -2.1746984, -1.4503986, -0.1922981, 0.1926485
7: -1.8795152, -0.9334493, -1.8795151, -0.9333362, -0.3937425, 0.3935792
8: -3.9261580, -2.1343107, -3.9261782, -2.1342294, -0.7547425, 0.7546409
9: -4.5590892, -3.0845289, -4.5591240, -3.0845282, -0.5455238, 0.5455705

Time for backsubstitution: 5.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 288
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3454
type: B, layer: 1, pos: 3085
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 303
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2109
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 3534
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3268
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 2068
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2308
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3012
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3177
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2256
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2343
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2487
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 3163
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2309
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 3239
type: B, layer: 1, pos: 3524

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 375

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0249256, upper bound: 0.0248726
time: 144.30 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0249261, upper bound: 0.0250285
time: 194.68 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.6969855, -2.2370288, -3.6959128, -2.2370310, -0.6445817, 0.6433083
1: -3.8972290, -2.2362113, -3.8942282, -2.2368879, -0.7018343, 0.6983939
2: -2.2246253, -1.6141737, -2.2233362, -1.6189114, -0.4837910, 0.4872735
3: 0.4529106, 0.7559570, 0.4531265, 0.7521139, -0.1811627, 0.1853166
4: -3.0148017, -2.2257228, -3.0147297, -2.2268517, -0.3146244, 0.3163450
5: 0.8822080, 1.2155695, 0.8825113, 1.2126901, -0.1495328, 0.1526744
6: -2.1780155, -1.4432694, -2.1778345, -1.4503435, -0.1928996, 0.2027067
7: -1.8819064, -0.9320374, -1.8795151, -0.9322343, -0.3980790, 0.3947266
8: -3.9282889, -2.1335258, -3.9263406, -2.1335738, -0.7578506, 0.7547875
9: -4.5575380, -3.0847268, -4.5575948, -3.0845234, -0.5455187, 0.5465472

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 288
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 2126
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 2215
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 3312
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 3084
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 3454
type: B, layer: 1, pos: 3085
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2267
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2491
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 2265
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 303
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2266
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2109
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2503
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 3534
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3268
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 3296
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2480
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 2068
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2271
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2493
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2308
type: B, layer: 1, pos: 2157
type: B, layer: 1, pos: 3012
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3177
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2256
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2343
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2487
type: B, layer: 1, pos: 2131
type: B, layer: 1, pos: 3163
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 2085
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 224
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2241
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2244
type: B, layer: 1, pos: 2309
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2624
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 3239
type: B, layer: 1, pos: 3524

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 375

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250268, upper bound: 0.0248702
time: 158.06 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250263, upper bound: 0.0250231
time: 275.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 439.68 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 439.68
Output dim: 3, lower bound: -0.0249256, upper bound: 0.0248726
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 439.68
Output dim: 3, lower bound: -0.0249261, upper bound: 0.0250285
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 439.68
Output dim: 3, lower bound: -0.0250268, upper bound: 0.0248702
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 439.68
Output dim: 3, lower bound: -0.0250263, upper bound: 0.0250231

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.6958771, -2.2372792, -3.6958818, -2.2372496, -0.6416498, 0.6431953
1: -3.8942282, -2.2385609, -3.8942282, -2.2383821, -0.6904417, 0.6965379
2: -2.2205400, -1.6189160, -2.2208421, -1.6189154, -0.4811829, 0.4769246
3: 0.4551718, 0.7521114, 0.4549535, 0.7521117, -0.1793925, 0.1736753
4: -3.0141337, -2.2268522, -3.0141878, -2.2268522, -0.3123500, 0.3145473
5: 0.8840587, 1.2126881, 0.8838926, 1.2126883, -0.1481956, 0.1393667
6: -2.1743150, -1.4504056, -2.1746945, -1.4503987, -0.1922944, 0.1906889
7: -1.8795151, -0.9334595, -1.8795151, -0.9333488, -0.3782646, 0.3935270
8: -3.9261069, -2.1343107, -3.9261179, -2.1342289, -0.7546055, 0.7546160
9: -4.5590901, -3.0845394, -4.5591240, -3.0845404, -0.5398256, 0.5455636

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 288
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3454
type: A, layer: 1, pos: 3085
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 303
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 2109
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 3534
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3268
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 2068
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 2308
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 3012
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3177
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2256
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 2343
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2487
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 3163
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2309
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 3239
type: A, layer: 1, pos: 3524

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2572

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0249078, upper bound: 0.0249888
time: 166.41 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0249110, upper bound: 0.0250145
time: 12.34 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.6948428, -2.2384205, -3.6922617, -2.2386608, -0.6404272, 0.6379517
1: -3.8972285, -2.2410038, -3.8930018, -2.2425132, -0.6963593, 0.6924101
2: -2.2201660, -1.6142106, -2.2178938, -1.6202095, -0.4780008, 0.4817802
3: 0.4572811, 0.7558444, 0.4582527, 0.7504290, -0.1750579, 0.1802336
4: -3.0147047, -2.2275310, -3.0141389, -2.2289701, -0.3124814, 0.3140266
5: 0.8886810, 1.2155448, 0.8900965, 1.2109216, -0.1413193, 0.1450903
6: -2.1766031, -1.4434175, -2.1761789, -1.4510235, -0.1908163, 0.2009274
7: -1.8818986, -0.9439237, -1.8766413, -0.9459597, -0.3844942, 0.3801947
8: -3.9276354, -2.1335335, -3.9253709, -2.1331992, -0.7572218, 0.7537189
9: -4.5575352, -3.0888476, -4.5565386, -3.0893631, -0.5406980, 0.5413315

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 288
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3085
type: A, layer: 1, pos: 3454
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 303
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 2109
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 3534
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3268
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2068
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 2308
type: A, layer: 1, pos: 3012
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3177
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2256
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 2343
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2487
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3163
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2309
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 3239
type: A, layer: 1, pos: 3524

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2572

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250090, upper bound: 0.0248367
time: 47.96 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250117, upper bound: 0.0248613
time: 12.73 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.6969831, -2.2370324, -3.6959109, -2.2370355, -0.6430055, 0.6433033
1: -3.8972290, -2.2362227, -3.8942282, -2.2369013, -0.6955146, 0.6983720
2: -2.2246208, -1.6141734, -2.2233307, -1.6189115, -0.4837824, 0.4827734
3: 0.4529107, 0.7559569, 0.4531265, 0.7521138, -0.1811603, 0.1793771
4: -3.0148017, -2.2257230, -3.0147293, -2.2268519, -0.3124910, 0.3163444
5: 0.8822109, 1.2155694, 0.8825147, 1.2126901, -0.1495296, 0.1436742
6: -2.1780124, -1.4432691, -2.1778312, -1.4503437, -0.1928958, 0.2007471
7: -1.8819063, -0.9320481, -1.8795151, -0.9322473, -0.3826028, 0.3946745
8: -3.9282384, -2.1335258, -3.9262800, -2.1335738, -0.7577137, 0.7547626
9: -4.5575380, -3.0847373, -4.5575948, -3.0845361, -0.5398206, 0.5465403

Time for backsubstitution: 5.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 288
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 2126
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 2215
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2181
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3312
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 3084
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 3085
type: A, layer: 1, pos: 3454
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2267
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2491
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 2265
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 303
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2266
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 2109
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 2503
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 3534
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3268
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3296
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 2480
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2068
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2271
type: A, layer: 1, pos: 2157
type: A, layer: 1, pos: 2493
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 2308
type: A, layer: 1, pos: 3012
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 3177
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2066
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2256
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 2343
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2487
type: A, layer: 1, pos: 2131
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 3163
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 2116
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2085
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 224
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2241
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2244
type: A, layer: 1, pos: 2309
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2624
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2819
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 3239
type: A, layer: 1, pos: 3524

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2572

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250096, upper bound: 0.0249936
time: 26.89 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250118, upper bound: 0.0250124
time: 281.94 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 314.83 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 314.83
Output dim: 3, lower bound: -0.0249078, upper bound: 0.0249888
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 314.83
Output dim: 3, lower bound: -0.0249110, upper bound: 0.0250145
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 314.83
Output dim: 3, lower bound: -0.0250090, upper bound: 0.0248367
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 314.83
Output dim: 3, lower bound: -0.0250117, upper bound: 0.0248613
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 314.83
Output dim: 3, lower bound: -0.0250096, upper bound: 0.0249936
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 314.83
Output dim: 3, lower bound: -0.0250118, upper bound: 0.0250124

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 43.22 + 1804.14 = 1847.36 seconds
