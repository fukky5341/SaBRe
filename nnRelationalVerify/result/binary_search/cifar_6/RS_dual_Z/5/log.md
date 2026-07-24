## Execution arguments:
Dataset: Dataset.CIFAR10
Network: onnx/cifar10_conv_exp.onnx
Epsilon: 0.03125
Initial delta epsilon: 8
Time budget: 18000 seconds
Threshold: 0.1215620163
Search space: {k/256.0 | k = 1, 2, ..., 8}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.7101218, 0.7101218)
1: (-3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9608312, 2.9608312)
2: (0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2670860, 0.2670860)
3: (-2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.2302514, 1.2302513)
4: (-2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.9083476, 0.9083476)
5: (-2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.3227956, 1.3227956)
6: (-6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.1522996, 1.1522996)
7: (-2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7862353, 1.7862351)
8: (-2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6292872, 2.6292868)
9: (-3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8981837, 1.8981836)

## BASE Result
execution time: IAR + LP analysis = 4.97 + 20.21 = 25.17 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 17974.83 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=8, k_mid=4, eps_mid=0.0156250, abs_max=0.8659994602203369
rel_dist={4: [-0.1637425378646875, 0.1637597121428953]}

## Binary search (step 1) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=3, k_mid=2, eps_mid=0.0078125, abs_max=0.8448255062103271
rel_dist={4: [-0.1216752063496438, 0.12168907463758383]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=1, k_mid=1, eps_mid=0.0039062, abs_max=0.8342382907867432
rel_dist={4: [-0.10046325656480493, 0.10048674767823251]}

## Binary Search Result
Binary search time: 290.44 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Relational Split (RS_dual_Z) starts
Time budget: 17684.39 seconds

## Binary search (step 0) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 3129

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846283, upper bound: 0.1846633
time: 140.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846629, upper bound: 0.1846295
time: 191.92 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 332.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 332.49
Output dim: 4, lower bound: -0.1846283, upper bound: 0.1846633
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 332.49
Output dim: 4, lower bound: -0.1846629, upper bound: 0.1846295

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6929235, 0.6929235
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9257360, 2.9257362
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477250, 0.2477250
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981555, 1.1981556
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765844, 0.8765844
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2879291, 1.2879292
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0504236, 1.0504236
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221980, 1.7221981
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6085114, 2.6085114
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584718, 1.8584719

Time for backsubstitution: 4.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845733, upper bound: 0.1846338
time: 260.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845998, upper bound: 0.1846081
time: 56.19 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6929235, 0.6929234
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9257360, 2.9257362
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477250, 0.2477250
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981555, 1.1981556
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765844, 0.8765844
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2879291, 1.2879292
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0504236, 1.0504236
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221980, 1.7221982
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6085114, 2.6085114
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584718, 1.8584719

Time for backsubstitution: 4.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846093, upper bound: 0.1845957
time: 48.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846347, upper bound: 0.1845714
time: 148.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 201.57 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 201.57
Output dim: 4, lower bound: -0.1845733, upper bound: 0.1846338
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 201.57
Output dim: 4, lower bound: -0.1845998, upper bound: 0.1846081
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 201.57
Output dim: 4, lower bound: -0.1846093, upper bound: 0.1845957
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 201.57
Output dim: 4, lower bound: -0.1846347, upper bound: 0.1845714

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6929227, 0.6929228
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9257360, 2.9257357
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477242, 0.2477242
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981553, 1.1981554
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765837, 0.8765836
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2879287, 1.2879287
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0504234, 1.0504235
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221973, 1.7221969
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6085114, 2.6085114
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584708, 1.8584712

Time for backsubstitution: 4.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845035, upper bound: 0.1846173
time: 139.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845495, upper bound: 0.1845636
time: 118.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6929228, 0.6929228
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9257360, 2.9257357
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477242, 0.2477242
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981553, 1.1981553
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765837, 0.8765836
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2879287, 1.2879288
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0504236, 1.0504234
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221968, 1.7221972
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6085114, 2.6085114
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584710, 1.8584709

Time for backsubstitution: 4.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845287, upper bound: 0.1845828
time: 424.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845734, upper bound: 0.1845397
time: 401.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6929228, 0.6929227
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9257360, 2.9257357
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477242, 0.2477242
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981553, 1.1981554
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765837, 0.8765836
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2879287, 1.2879287
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0504234, 1.0504234
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221973, 1.7221971
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6085114, 2.6085114
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584708, 1.8584709

Time for backsubstitution: 4.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 3109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845421, upper bound: 0.1845263
time: 664.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845866, upper bound: 0.1845307
time: 42.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6929228, 0.6929227
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9257360, 2.9257357
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477242, 0.2477242
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981553, 1.1981553
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765835, 0.8765836
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2879287, 1.2879288
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0504236, 1.0504234
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221968, 1.7221973
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6085114, 2.6085114
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584713, 1.8584709

Time for backsubstitution: 4.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845645, upper bound: 0.1845539
time: 204.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846101, upper bound: 0.1845011
time: 243.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 453.21 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 453.21
Output dim: 4, lower bound: -0.1845035, upper bound: 0.1846173
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 453.21
Output dim: 4, lower bound: -0.1845495, upper bound: 0.1845636
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 453.21
Output dim: 4, lower bound: -0.1845287, upper bound: 0.1845828
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 453.21
Output dim: 4, lower bound: -0.1845734, upper bound: 0.1845397
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 453.21
Output dim: 4, lower bound: -0.1845421, upper bound: 0.1845263
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 453.21
Output dim: 4, lower bound: -0.1845866, upper bound: 0.1845307
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 453.21
Output dim: 4, lower bound: -0.1845645, upper bound: 0.1845539
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 453.21
Output dim: 4, lower bound: -0.1846101, upper bound: 0.1845011

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6929228, 0.6929228
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9257336, 2.9257333
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477236, 0.2477237
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981552, 1.1981550
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765832, 0.8765832
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2879287, 1.2879287
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0504234, 1.0504233
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221959, 1.7221955
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6085119, 2.6085119
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584683, 1.8584685

Time for backsubstitution: 4.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2646

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1843896, upper bound: 0.1845062
time: 18.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1843901, upper bound: 0.1845022
time: 14.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6929228, 0.6929228
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9257333, 2.9257333
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477236, 0.2477236
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981549, 1.1981552
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765832, 0.8765832
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2879287, 1.2879287
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0504231, 1.0504235
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221959, 1.7221954
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6085119, 2.6085119
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584683, 1.8584685

Time for backsubstitution: 4.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 2646

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844353, upper bound: 0.1844562
time: 19.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844353, upper bound: 0.1844477
time: 202.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6929228, 0.6929228
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9257333, 2.9257333
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477236, 0.2477237
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981552, 1.1981549
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765832, 0.8765833
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2879287, 1.2879288
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0504236, 1.0504231
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221954, 1.7221957
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6085119, 2.6085119
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584685, 1.8584681

Time for backsubstitution: 4.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2646

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844142, upper bound: 0.1844731
time: 83.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844151, upper bound: 0.1844726
time: 173.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6929228, 0.6929228
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9257333, 2.9257333
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477236, 0.2477236
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981549, 1.1981552
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765832, 0.8765833
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2879287, 1.2879288
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0504234, 1.0504234
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221954, 1.7221956
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6085119, 2.6085119
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584683, 1.8584683

Time for backsubstitution: 4.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 2646

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844618, upper bound: 0.1844358
time: 18.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844617, upper bound: 0.1844325
time: 22.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6929228, 0.6929228
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9257333, 2.9257333
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477236, 0.2477237
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981552, 1.1981550
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765832, 0.8765833
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2879287, 1.2879287
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0504234, 1.0504231
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221959, 1.7221956
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6085119, 2.6085119
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584683, 1.8584685

Time for backsubstitution: 4.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 2646

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844271, upper bound: 0.1844705
time: 19.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844247, upper bound: 0.1844617
time: 165.63 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 189.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 189.93
Output dim: 4, lower bound: -0.1843896, upper bound: 0.1845062
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 189.93
Output dim: 4, lower bound: -0.1843901, upper bound: 0.1845022
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 189.93
Output dim: 4, lower bound: -0.1844353, upper bound: 0.1844562
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 189.93
Output dim: 4, lower bound: -0.1844353, upper bound: 0.1844477
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 189.93
Output dim: 4, lower bound: -0.1844142, upper bound: 0.1844731
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 189.93
Output dim: 4, lower bound: -0.1844151, upper bound: 0.1844726
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 189.93
Output dim: 4, lower bound: -0.1844618, upper bound: 0.1844358
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 189.93
Output dim: 4, lower bound: -0.1844617, upper bound: 0.1844325
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 189.93
Output dim: 4, lower bound: -0.1844271, upper bound: 0.1844705
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 189.93
Output dim: 4, lower bound: -0.1844247, upper bound: 0.1844617
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 189.93
Output dim: 4, lower bound: -0.1845866, upper bound: 0.1845307
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 189.93
Output dim: 4, lower bound: -0.1845645, upper bound: 0.1845539
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 189.93
Output dim: 4, lower bound: -0.1846101, upper bound: 0.1845011
Binary search (step 0): status=Status.UNKNOWN, k_low=2, k_high=8, k_mid=5, eps_mid=0.0195312, abs_max=0.8765864372253418
rel_dist={4: [-0.18467027294666494, 0.18467521355340155]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 3129

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427412, upper bound: 0.1427660
time: 251.53 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427641, upper bound: 0.1427471
time: 175.68 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 427.39 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 427.39
Output dim: 4, lower bound: -0.1427412, upper bound: 0.1427660
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 427.39
Output dim: 4, lower bound: -0.1427641, upper bound: 0.1427471

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814604, 0.6814605
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023395, 2.9023397
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348181, 0.2348181
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767586, 1.1767586
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554102, 0.8554102
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646849, 1.2646849
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825076, 0.9825077
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795092, 1.6795089
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946641, 2.5946641
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319975, 1.8319974

Time for backsubstitution: 4.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427068, upper bound: 0.1427503
time: 196.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427225, upper bound: 0.1427303
time: 339.39 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814606, 0.6814605
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023395, 2.9023397
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348181, 0.2348181
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767586, 1.1767586
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554102, 0.8554102
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646849, 1.2646849
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825076, 0.9825076
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795087, 1.6795089
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946641, 2.5946641
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319975, 1.8319974

Time for backsubstitution: 4.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427285, upper bound: 0.1427279
time: 70.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427468, upper bound: 0.1427124
time: 341.72 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 417.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 417.10
Output dim: 4, lower bound: -0.1427068, upper bound: 0.1427503
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 417.10
Output dim: 4, lower bound: -0.1427225, upper bound: 0.1427303
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 417.10
Output dim: 4, lower bound: -0.1427285, upper bound: 0.1427279
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 417.10
Output dim: 4, lower bound: -0.1427468, upper bound: 0.1427124

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814597, 0.6814598
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023395, 2.9023392
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348173, 0.2348173
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767584, 1.1767584
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554095, 0.8554094
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646846, 1.2646844
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825074, 0.9825075
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795080, 1.6795077
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946641, 2.5946641
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319966, 1.8319967

Time for backsubstitution: 4.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 3109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426635, upper bound: 0.1427344
time: 61.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426881, upper bound: 0.1427133
time: 19.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814597, 0.6814598
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023395, 2.9023392
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348173, 0.2348173
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767584, 1.1767584
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554095, 0.8554095
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646846, 1.2646846
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825074, 0.9825074
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795080, 1.6795079
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946641, 2.5946641
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319968, 1.8319964

Time for backsubstitution: 4.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426814, upper bound: 0.1427218
time: 30.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427015, upper bound: 0.1426909
time: 160.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814597, 0.6814598
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023395, 2.9023392
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348173, 0.2348173
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767584, 1.1767584
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554095, 0.8554094
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646846, 1.2646844
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825074, 0.9825074
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795080, 1.6795077
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946641, 2.5946641
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319966, 1.8319967

Time for backsubstitution: 4.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 3109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426837, upper bound: 0.1427209
time: 12.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427111, upper bound: 0.1426860
time: 188.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814597, 0.6814597
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023395, 2.9023392
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348173, 0.2348173
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767584, 1.1767584
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554093, 0.8554095
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646846, 1.2646846
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825074, 0.9825074
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795075, 1.6795079
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946641, 2.5946641
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319968, 1.8319964

Time for backsubstitution: 4.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 3109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427032, upper bound: 0.1426963
time: 51.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427302, upper bound: 0.1426650
time: 224.03 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 280.30 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 280.30
Output dim: 4, lower bound: -0.1426635, upper bound: 0.1427344
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 280.30
Output dim: 4, lower bound: -0.1426881, upper bound: 0.1427133
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 280.30
Output dim: 4, lower bound: -0.1426814, upper bound: 0.1427218
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 280.30
Output dim: 4, lower bound: -0.1427015, upper bound: 0.1426909
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 280.30
Output dim: 4, lower bound: -0.1426837, upper bound: 0.1427209
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 280.30
Output dim: 4, lower bound: -0.1427111, upper bound: 0.1426860
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 280.30
Output dim: 4, lower bound: -0.1427032, upper bound: 0.1426963
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 280.30
Output dim: 4, lower bound: -0.1427302, upper bound: 0.1426650

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814597, 0.6814598
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023368, 2.9023368
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348167, 0.2348168
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767583, 1.1767581
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554093, 0.8554091
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646846, 1.2646844
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825074, 0.9825072
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795065, 1.6795062
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946641, 2.5946641
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319938, 1.8319941

Time for backsubstitution: 4.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 2646

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425801, upper bound: 0.1426502
time: 424.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425788, upper bound: 0.1426550
time: 20.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814597, 0.6814598
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023368, 2.9023368
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348168, 0.2348167
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767581, 1.1767582
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554093, 0.8554091
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646846, 1.2646844
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825072, 0.9825073
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795065, 1.6795062
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946641, 2.5946641
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319938, 1.8319941

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 2646

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426043, upper bound: 0.1426267
time: 25.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426050, upper bound: 0.1426043
time: 248.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814597, 0.6814598
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023368, 2.9023368
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348167, 0.2348168
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767583, 1.1767581
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554093, 0.8554091
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646846, 1.2646846
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825074, 0.9825072
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795061, 1.6795063
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946641, 2.5946641
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319941, 1.8319941

Time for backsubstitution: 4.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 2646

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425942, upper bound: 0.1426339
time: 276.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425952, upper bound: 0.1426293
time: 232.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814598, 0.6814598
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023368, 2.9023368
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348168, 0.2348167
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767581, 1.1767582
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554093, 0.8554091
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646846, 1.2646846
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825072, 0.9825073
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795061, 1.6795063
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946641, 2.5946641
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319941, 1.8319941

Time for backsubstitution: 4.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 2646

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426228, upper bound: 0.1426035
time: 107.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426225, upper bound: 0.1425779
time: 177.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814598, 0.6814598
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023368, 2.9023368
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348167, 0.2348168
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767583, 1.1767581
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554093, 0.8554091
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646846, 1.2646844
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825074, 0.9825072
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795065, 1.6795063
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946641, 2.5946641
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319938, 1.8319941

Time for backsubstitution: 4.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 2646

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425955, upper bound: 0.1426343
time: 14.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425981, upper bound: 0.1426257
time: 385.97 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 405.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 405.22
Output dim: 4, lower bound: -0.1425801, upper bound: 0.1426502
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 405.22
Output dim: 4, lower bound: -0.1425788, upper bound: 0.1426550
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 405.22
Output dim: 4, lower bound: -0.1426043, upper bound: 0.1426267
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 405.22
Output dim: 4, lower bound: -0.1426050, upper bound: 0.1426043
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 405.22
Output dim: 4, lower bound: -0.1425942, upper bound: 0.1426339
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 405.22
Output dim: 4, lower bound: -0.1425952, upper bound: 0.1426293
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 405.22
Output dim: 4, lower bound: -0.1426228, upper bound: 0.1426035
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 405.22
Output dim: 4, lower bound: -0.1426225, upper bound: 0.1425779
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 405.22
Output dim: 4, lower bound: -0.1425955, upper bound: 0.1426343
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 405.22
Output dim: 4, lower bound: -0.1425981, upper bound: 0.1426257
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 405.22
Output dim: 4, lower bound: -0.1427111, upper bound: 0.1426860
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 405.22
Output dim: 4, lower bound: -0.1427032, upper bound: 0.1426963
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 405.22
Output dim: 4, lower bound: -0.1427302, upper bound: 0.1426650
Binary search (step 1): status=Status.UNKNOWN, k_low=2, k_high=4, k_mid=3, eps_mid=0.0117188, abs_max=0.8554123640060425
rel_dist={4: [-0.1427653512292979, 0.1427739341779526]}

## Binary search (step 2) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 3129

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216559, upper bound: 0.1216925
time: 11.53 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216686, upper bound: 0.1216577
time: 496.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 508.18 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 508.18
Output dim: 4, lower bound: -0.1216559, upper bound: 0.1216925
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 508.18
Output dim: 4, lower bound: -0.1216686, upper bound: 0.1216577

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6757290, 0.6757290
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8906412, 2.8906415
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283646, 0.2283646
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1660601, 1.1660602
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8448232, 0.8448232
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2530627, 1.2530628
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9485497, 0.9485497
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6581645, 1.6581643
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877404, 2.5877399
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8187603, 1.8187604

Time for backsubstitution: 4.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216228, upper bound: 0.1216774
time: 16.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216403, upper bound: 0.1216561
time: 101.76 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6757291, 0.6757290
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8906412, 2.8906415
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283646, 0.2283646
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1660601, 1.1660602
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8448232, 0.8448232
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2530627, 1.2530628
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9485497, 0.9485497
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6581645, 1.6581643
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877404, 2.5877399
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8187603, 1.8187604

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 2439

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216446, upper bound: 0.1216584
time: 237.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216569, upper bound: 0.1216334
time: 164.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 406.66 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 406.66
Output dim: 4, lower bound: -0.1216228, upper bound: 0.1216774
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 406.66
Output dim: 4, lower bound: -0.1216403, upper bound: 0.1216561
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 406.66
Output dim: 4, lower bound: -0.1216446, upper bound: 0.1216584
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 406.66
Output dim: 4, lower bound: -0.1216569, upper bound: 0.1216334

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6757283, 0.6757283
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8906407, 2.8906410
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283638, 0.2283638
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1660599, 1.1660600
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8448225, 0.8448224
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2530624, 1.2530624
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9485494, 0.9485494
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6581633, 1.6581631
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877404, 2.5877399
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8187593, 1.8187594

Time for backsubstitution: 4.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 3109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216053, upper bound: 0.1216563
time: 35.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216231, upper bound: 0.1216459
time: 29.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6757283, 0.6757283
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8906407, 2.8906410
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283638, 0.2283638
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1660599, 1.1660600
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8448225, 0.8448224
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2530624, 1.2530624
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9485494, 0.9485494
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6581633, 1.6581632
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877404, 2.5877399
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8187593, 1.8187594

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 3109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216183, upper bound: 0.1216457
time: 193.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216335, upper bound: 0.1216260
time: 39.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6757283, 0.6757283
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8906407, 2.8906410
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283638, 0.2283638
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1660599, 1.1660600
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8448225, 0.8448224
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2530624, 1.2530624
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9485494, 0.9485494
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6581633, 1.6581631
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877404, 2.5877399
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8187593, 1.8187594

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 3109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216170, upper bound: 0.1216386
time: 47.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216331, upper bound: 0.1216214
time: 259.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6757283, 0.6757283
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8906407, 2.8906410
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283638, 0.2283638
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1660599, 1.1660600
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8448225, 0.8448224
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2530624, 1.2530624
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9485494, 0.9485494
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6581633, 1.6581632
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877404, 2.5877399
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8187593, 1.8187594

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 3109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216298, upper bound: 0.1216050
time: 168.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216429, upper bound: 0.1216209
time: 14.61 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 187.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 187.25
Output dim: 4, lower bound: -0.1216053, upper bound: 0.1216563
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 187.25
Output dim: 4, lower bound: -0.1216231, upper bound: 0.1216459
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 187.25
Output dim: 4, lower bound: -0.1216183, upper bound: 0.1216457
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 187.25
Output dim: 4, lower bound: -0.1216335, upper bound: 0.1216260
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 187.25
Output dim: 4, lower bound: -0.1216170, upper bound: 0.1216386
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 187.25
Output dim: 4, lower bound: -0.1216331, upper bound: 0.1216214
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 187.25
Output dim: 4, lower bound: -0.1216298, upper bound: 0.1216050
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 187.25
Output dim: 4, lower bound: -0.1216429, upper bound: 0.1216209

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6757283, 0.6757283
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8906386, 2.8906386
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283633, 0.2283633
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1660595, 1.1660596
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8448220, 0.8448220
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2530624, 1.2530624
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9485492, 0.9485492
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6581619, 1.6581616
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877404, 2.5877404
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8187568, 1.8187568

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 2646

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215368, upper bound: 0.1215950
time: 19.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215323, upper bound: 0.1215915
time: 154.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6757283, 0.6757283
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8906386, 2.8906386
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283633, 0.2283633
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1660595, 1.1660597
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8448220, 0.8448220
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2530624, 1.2530624
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9485492, 0.9485493
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6581619, 1.6581616
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877404, 2.5877404
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8187568, 1.8187571

Time for backsubstitution: 4.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 2646

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215633, upper bound: 0.1215703
time: 225.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215561, upper bound: 0.1215674
time: 162.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6757283, 0.6757283
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8906386, 2.8906386
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283633, 0.2283633
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1660597, 1.1660596
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8448220, 0.8448222
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2530624, 1.2530624
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9485492, 0.9485492
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6581614, 1.6581616
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877404, 2.5877404
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8187568, 1.8187566

Time for backsubstitution: 4.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 2646

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215418, upper bound: 0.1215855
time: 17.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1215425, upper bound: 0.1215359
time: 467.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6757284, 0.6757283
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8906386, 2.8906386
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283633, 0.2283633
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1660595, 1.1660597
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8448220, 0.8448222
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2530624, 1.2530624
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9485492, 0.9485493
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6581614, 1.6581616
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877404, 2.5877404
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8187568, 1.8187568

Time for backsubstitution: 4.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 2646

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215668, upper bound: 0.1215553
time: 86.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215644, upper bound: 0.1215537
time: 389.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6757284, 0.6757283
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8906386, 2.8906386
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283633, 0.2283633
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1660597, 1.1660596
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8448220, 0.8448222
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2530624, 1.2530624
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9485492, 0.9485492
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6581619, 1.6581616
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877404, 2.5877404
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8187568, 1.8187568

Time for backsubstitution: 4.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 2646

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215473, upper bound: 0.1215743
time: 194.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215538, upper bound: 0.1215839
time: 15.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6757284, 0.6757283
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8906386, 2.8906386
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283633, 0.2283633
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1660595, 1.1660597
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8448220, 0.8448220
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2530624, 1.2530624
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9485492, 0.9485493
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6581619, 1.6581616
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877404, 2.5877404
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8187568, 1.8187568

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 2646

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215727, upper bound: 0.1215359
time: 164.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215634, upper bound: 0.1215624
time: 167.13 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 335.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 335.64
Output dim: 4, lower bound: -0.1215368, upper bound: 0.1215950
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 335.64
Output dim: 4, lower bound: -0.1215323, upper bound: 0.1215915
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 335.64
Output dim: 4, lower bound: -0.1215633, upper bound: 0.1215703
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 335.64
Output dim: 4, lower bound: -0.1215561, upper bound: 0.1215674
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 335.64
Output dim: 4, lower bound: -0.1215418, upper bound: 0.1215855
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 335.64
Output dim: 4, lower bound: -0.1215425, upper bound: 0.1215359
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 335.64
Output dim: 4, lower bound: -0.1215668, upper bound: 0.1215553
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 335.64
Output dim: 4, lower bound: -0.1215644, upper bound: 0.1215537
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 335.64
Output dim: 4, lower bound: -0.1215473, upper bound: 0.1215743
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 335.64
Output dim: 4, lower bound: -0.1215538, upper bound: 0.1215839
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 335.64
Output dim: 4, lower bound: -0.1215727, upper bound: 0.1215359
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 335.64
Output dim: 4, lower bound: -0.1215634, upper bound: 0.1215624
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 335.64
Output dim: 4, lower bound: -0.1216298, upper bound: 0.1216050
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 335.64
Output dim: 4, lower bound: -0.1216429, upper bound: 0.1216209
Binary search (step 2): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=0.8448255062103271
rel_dist={4: [-0.1216752063496438, 0.12168907463758383]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 12316.40 seconds
