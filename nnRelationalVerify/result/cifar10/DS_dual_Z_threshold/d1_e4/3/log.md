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
execution time: IAR + RelationalAnalysis = 7.72 + 35.26 = 42.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0250319, upper bound: 0.0250374

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2308
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3163
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3268
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3524

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3401

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250278, upper bound: 0.0250280
time: 194.01 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250354, upper bound: 0.0250320
time: 12.03 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 206.12 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 206.12
Output dim: 3, lower bound: -0.0250278, upper bound: 0.0250280
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 206.12
Output dim: 3, lower bound: -0.0250354, upper bound: 0.0250320

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.6959207, -2.2369895, -3.6959207, -2.2369895, -0.6433077, 0.6433214
1: -3.8942285, -2.2368288, -3.8942285, -2.2368288, -0.6984020, 0.6984122
2: -2.2234130, -1.6189097, -2.2234130, -1.6189097, -0.4836495, 0.4836577
3: 0.4531021, 0.7521143, 0.4531021, 0.7521143, -0.1814679, 0.1814688
4: -3.0147476, -2.2268522, -3.0147476, -2.2268522, -0.3151326, 0.3151265
5: 0.8824778, 1.2126909, 0.8824778, 1.2126909, -0.1497163, 0.1497224
6: -2.1779187, -1.4503413, -2.1779187, -1.4503413, -0.1959779, 0.1959729
7: -1.8795151, -0.9322217, -1.8795151, -0.9322217, -0.3951584, 0.3951549
8: -3.9263482, -2.1335392, -3.9263482, -2.1335392, -0.7561275, 0.7561163
9: -4.5595870, -3.0845213, -4.5595870, -3.0845213, -0.5464227, 0.5464217

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2308
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3163
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3268
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3524

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 463

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250244, upper bound: 0.0250340
time: 165.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250272, upper bound: 0.0250310
time: 186.37 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.6959207, -2.2369895, -3.6959207, -2.2369895, -0.6433214, 0.6433076
1: -3.8942285, -2.2368288, -3.8942285, -2.2368288, -0.6984122, 0.6984019
2: -2.2234130, -1.6189097, -2.2234130, -1.6189097, -0.4836577, 0.4836495
3: 0.4531021, 0.7521143, 0.4531021, 0.7521143, -0.1814688, 0.1814679
4: -3.0147476, -2.2268522, -3.0147476, -2.2268522, -0.3151265, 0.3151326
5: 0.8824778, 1.2126909, 0.8824778, 1.2126909, -0.1497224, 0.1497163
6: -2.1779187, -1.4503413, -2.1779187, -1.4503413, -0.1959729, 0.1959778
7: -1.8795151, -0.9322217, -1.8795151, -0.9322217, -0.3951548, 0.3951585
8: -3.9263482, -2.1335392, -3.9263482, -2.1335392, -0.7561163, 0.7561275
9: -4.5595870, -3.0845213, -4.5595870, -3.0845213, -0.5464217, 0.5464226

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2308
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3163
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3268
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3524

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 463

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250310, upper bound: 0.0250259
time: 107.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250323, upper bound: 0.0250295
time: 13.41 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 127.16 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 127.16
Output dim: 3, lower bound: -0.0250244, upper bound: 0.0250340
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 127.16
Output dim: 3, lower bound: -0.0250272, upper bound: 0.0250310
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 127.16
Output dim: 3, lower bound: -0.0250310, upper bound: 0.0250259
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 127.16
Output dim: 3, lower bound: -0.0250323, upper bound: 0.0250295

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.6959207, -2.2369895, -3.6959207, -2.2369895, -0.6433069, 0.6433209
1: -3.8942285, -2.2368288, -3.8942285, -2.2368288, -0.6984020, 0.6984123
2: -2.2234130, -1.6189097, -2.2234130, -1.6189097, -0.4836478, 0.4836560
3: 0.4531021, 0.7521143, 0.4531021, 0.7521143, -0.1814678, 0.1814688
4: -3.0147476, -2.2268522, -3.0147476, -2.2268522, -0.3151299, 0.3151237
5: 0.8824778, 1.2126909, 0.8824778, 1.2126909, -0.1497165, 0.1497226
6: -2.1779187, -1.4503413, -2.1779187, -1.4503413, -0.1959784, 0.1959736
7: -1.8795151, -0.9322217, -1.8795151, -0.9322217, -0.3951583, 0.3951547
8: -3.9263482, -2.1335392, -3.9263482, -2.1335392, -0.7561277, 0.7561166
9: -4.5595870, -3.0845213, -4.5595870, -3.0845213, -0.5464243, 0.5464235

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2308
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3163
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3268
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3524

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 462

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250237, upper bound: 0.0250345
time: 161.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250257, upper bound: 0.0250267
time: 135.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.6959207, -2.2369895, -3.6959207, -2.2369895, -0.6433070, 0.6433207
1: -3.8942285, -2.2368288, -3.8942285, -2.2368288, -0.6984020, 0.6984123
2: -2.2234130, -1.6189097, -2.2234130, -1.6189097, -0.4836478, 0.4836560
3: 0.4531021, 0.7521143, 0.4531021, 0.7521143, -0.1814678, 0.1814688
4: -3.0147476, -2.2268522, -3.0147476, -2.2268522, -0.3151298, 0.3151238
5: 0.8824778, 1.2126909, 0.8824778, 1.2126909, -0.1497165, 0.1497226
6: -2.1779187, -1.4503413, -2.1779187, -1.4503413, -0.1959785, 0.1959735
7: -1.8795151, -0.9322217, -1.8795151, -0.9322217, -0.3951582, 0.3951547
8: -3.9263482, -2.1335392, -3.9263482, -2.1335392, -0.7561277, 0.7561165
9: -4.5595870, -3.0845213, -4.5595870, -3.0845213, -0.5464244, 0.5464234

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2308
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3163
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3268
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3524

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 462

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250268, upper bound: 0.0250231
time: 169.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250264, upper bound: 0.0250230
time: 227.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.6959207, -2.2369895, -3.6959207, -2.2369895, -0.6433207, 0.6433070
1: -3.8942285, -2.2368288, -3.8942285, -2.2368288, -0.6984124, 0.6984020
2: -2.2234130, -1.6189097, -2.2234130, -1.6189097, -0.4836560, 0.4836478
3: 0.4531021, 0.7521143, 0.4531021, 0.7521143, -0.1814688, 0.1814678
4: -3.0147476, -2.2268522, -3.0147476, -2.2268522, -0.3151238, 0.3151298
5: 0.8824778, 1.2126909, 0.8824778, 1.2126909, -0.1497226, 0.1497165
6: -2.1779187, -1.4503413, -2.1779187, -1.4503413, -0.1959735, 0.1959785
7: -1.8795151, -0.9322217, -1.8795151, -0.9322217, -0.3951547, 0.3951582
8: -3.9263482, -2.1335392, -3.9263482, -2.1335392, -0.7561165, 0.7561277
9: -4.5595870, -3.0845213, -4.5595870, -3.0845213, -0.5464233, 0.5464244

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2109
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 775
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3012
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2487
type: DSZ, layer: 1, pos: 2068
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 2308
type: DSZ, layer: 1, pos: 2500
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 701
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3084
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2267
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2480
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3163
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 722
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3177
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3268
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 224
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 288
type: DSZ, layer: 1, pos: 303
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 584
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2085
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2244
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2265
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2271
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2343
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2491
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3296
type: DSZ, layer: 1, pos: 3312
type: DSZ, layer: 1, pos: 3454
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3524

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 462

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250294, upper bound: 0.0250270
time: 192.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0250328, upper bound: 0.0250280
time: 185.20 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 384.19 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 384.19
Output dim: 3, lower bound: -0.0250237, upper bound: 0.0250345
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 384.19
Output dim: 3, lower bound: -0.0250257, upper bound: 0.0250267
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 384.19
Output dim: 3, lower bound: -0.0250268, upper bound: 0.0250231
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 384.19
Output dim: 3, lower bound: -0.0250264, upper bound: 0.0250230
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 384.19
Output dim: 3, lower bound: -0.0250294, upper bound: 0.0250270
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 384.19
Output dim: 3, lower bound: -0.0250328, upper bound: 0.0250280
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 384.19
Output dim: 3, lower bound: -0.0250323, upper bound: 0.0250295

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 42.98 + 1782.19 = 1825.17 seconds
