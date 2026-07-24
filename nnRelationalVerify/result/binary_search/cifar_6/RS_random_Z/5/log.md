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
execution time: IAR + LP analysis = 5.75 + 20.14 = 25.89 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 17974.11 seconds, max iter: 100)

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
Binary search time: 290.75 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Relational Split (RS_random_Z) starts
Time budget: 17683.36 seconds

## Binary search (step 0) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2440

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2288

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846619, upper bound: 0.1846629
time: 341.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846611, upper bound: 0.1846678
time: 19.16 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 360.53 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 360.53
Output dim: 4, lower bound: -0.1846619, upper bound: 0.1846629
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 360.53
Output dim: 4, lower bound: -0.1846611, upper bound: 0.1846678

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6928694, 0.6928536
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9255862, 2.9256279
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477110, 0.2477174
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1980940, 1.1981109
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765790, 0.8765787
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2878695, 1.2878835
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0501966, 1.0502584
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221800, 1.7221811
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6084909, 2.6084919
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584635, 1.8584633

Time for backsubstitution: 4.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2183

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845462, upper bound: 0.1846452
time: 27.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846369, upper bound: 0.1845460
time: 22.07 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6928536, 0.6928694
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9256282, 2.9255860
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477174, 0.2477110
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981109, 1.1980939
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765786, 0.8765789
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2878835, 1.2878695
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0502586, 1.0501965
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221814, 1.7221799
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6084919, 2.6084905
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584633, 1.8584635

Time for backsubstitution: 4.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 3087

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2661

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846401, upper bound: 0.1846367
time: 72.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846403, upper bound: 0.1846373
time: 72.06 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 148.42 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 148.42
Output dim: 4, lower bound: -0.1845462, upper bound: 0.1846452
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 148.42
Output dim: 4, lower bound: -0.1846369, upper bound: 0.1845460
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 148.42
Output dim: 4, lower bound: -0.1846401, upper bound: 0.1846367
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 148.42
Output dim: 4, lower bound: -0.1846403, upper bound: 0.1846373

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6928580, 0.6928416
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9255781, 2.9256206
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477119, 0.2477182
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1980910, 1.1981080
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765798, 0.8765796
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2878622, 1.2878761
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0501894, 1.0502515
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221115, 1.7221102
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6084933, 2.6084945
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584573, 1.8584574

Time for backsubstitution: 4.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3027

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2665

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844479, upper bound: 0.1845295
time: 194.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844299, upper bound: 0.1845459
time: 201.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6928574, 0.6928422
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9255786, 2.9256201
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477119, 0.2477183
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1980910, 1.1981080
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765798, 0.8765796
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2878621, 1.2878761
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0501896, 1.0502512
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221091, 1.7221122
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6084933, 2.6084943
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584578, 1.8584570

Time for backsubstitution: 4.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 141

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2273

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846353, upper bound: 0.1845395
time: 132.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846286, upper bound: 0.1845403
time: 517.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6928536, 0.6928694
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9256201, 2.9255745
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477144, 0.2477092
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981045, 1.1980873
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765719, 0.8765723
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2878609, 1.2878429
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0502295, 1.0501578
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221557, 1.7221544
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6084907, 2.6084890
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584521, 1.8584502

Time for backsubstitution: 4.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2500

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845185, upper bound: 0.1845217
time: 17.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845185, upper bound: 0.1845223
time: 17.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6928536, 0.6928694
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9256167, 2.9255784
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477156, 0.2477080
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981045, 1.1980876
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765719, 0.8765723
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2878568, 1.2878469
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0502198, 1.0501676
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221555, 1.7221544
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6084907, 2.6084890
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584499, 1.8584523

Time for backsubstitution: 4.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 534

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3149

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846407, upper bound: 0.1846439
time: 226.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846407, upper bound: 0.1846425
time: 143.02 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 373.62 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 373.62
Output dim: 4, lower bound: -0.1844479, upper bound: 0.1845295
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 373.62
Output dim: 4, lower bound: -0.1844299, upper bound: 0.1845459
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 373.62
Output dim: 4, lower bound: -0.1846353, upper bound: 0.1845395
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 373.62
Output dim: 4, lower bound: -0.1846286, upper bound: 0.1845403
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 373.62
Output dim: 4, lower bound: -0.1845185, upper bound: 0.1845217
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 373.62
Output dim: 4, lower bound: -0.1845185, upper bound: 0.1845223
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 373.62
Output dim: 4, lower bound: -0.1846407, upper bound: 0.1846439
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 373.62
Output dim: 4, lower bound: -0.1846407, upper bound: 0.1846425

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6928579, 0.6928415
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9255781, 2.9256206
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477118, 0.2477182
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1980910, 1.1981080
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765798, 0.8765794
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2878619, 1.2878759
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0501890, 1.0502512
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221105, 1.7221098
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6084933, 2.6084945
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584573, 1.8584574

Time for backsubstitution: 4.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 3129

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 817

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844049, upper bound: 0.1845241
time: 21.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844390, upper bound: 0.1844791
time: 169.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6928579, 0.6928415
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9255781, 2.9256206
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477118, 0.2477182
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1980910, 1.1981080
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765798, 0.8765794
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2878619, 1.2878759
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0501890, 1.0502512
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221105, 1.7221098
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6084933, 2.6084945
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584573, 1.8584574

Time for backsubstitution: 4.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 784

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2314

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1843727, upper bound: 0.1845280
time: 15.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844099, upper bound: 0.1844886
time: 131.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6927910, 0.6927701
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9253998, 2.9254551
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477085, 0.2477145
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1980879, 1.1981047
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765608, 0.8765590
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2878573, 1.2878700
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0501318, 1.0501862
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7220457, 1.7220414
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6084766, 2.6084790
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584559, 1.8584548

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2479

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3086

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845284, upper bound: 0.1845381
time: 153.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846316, upper bound: 0.1844304
time: 24.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6927853, 0.6927793
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9254217, 2.9254417
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477086, 0.2477148
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1980879, 1.1981049
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765591, 0.8765606
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2878559, 1.2878716
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0501313, 1.0501933
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7220380, 1.7220501
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6084790, 2.6084776
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584557, 1.8584551

Time for backsubstitution: 4.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2653

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 581

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846143, upper bound: 0.1845333
time: 110.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846168, upper bound: 0.1845304
time: 23.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6928442, 0.6928642
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9255652, 2.9254873
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477116, 0.2477050
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981045, 1.1980871
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765442, 0.8765441
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2878573, 1.2878413
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0502236, 1.0501469
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221323, 1.7221419
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6084900, 2.6084881
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584459, 1.8584404

Time for backsubstitution: 4.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 808

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2316

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844332, upper bound: 0.1844393
time: 338.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844429, upper bound: 0.1844335
time: 68.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6928536, 0.6928599
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9255328, 2.9255745
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477144, 0.2477064
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981043, 1.1980873
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765719, 0.8765445
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2878609, 1.2878391
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0502188, 1.0501578
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221557, 1.7221309
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6084895, 2.6084890
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584423, 1.8584502

Time for backsubstitution: 4.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1842659, upper bound: 0.1845072
time: 211.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845146, upper bound: 0.1842683
time: 136.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6928536, 0.6928694
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9256167, 2.9255784
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477156, 0.2477080
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981045, 1.1980876
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765719, 0.8765723
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2878568, 1.2878469
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0502198, 1.0501676
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221555, 1.7221544
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6084907, 2.6084890
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584499, 1.8584523

Time for backsubstitution: 4.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2350

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2272

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846298, upper bound: 0.1846338
time: 18.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846259, upper bound: 0.1846307
time: 29.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6928536, 0.6928694
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9256167, 2.9255784
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2477156, 0.2477080
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1981045, 1.1980876
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8765719, 0.8765723
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2878568, 1.2878469
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0502198, 1.0501676
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7221555, 1.7221544
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6084907, 2.6084890
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584499, 1.8584523

Time for backsubstitution: 4.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2593

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2236

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846400, upper bound: 0.1846453
time: 40.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846400, upper bound: 0.1846395
time: 185.97 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 230.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 230.16
Output dim: 4, lower bound: -0.1844049, upper bound: 0.1845241
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 230.16
Output dim: 4, lower bound: -0.1844390, upper bound: 0.1844791
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 230.16
Output dim: 4, lower bound: -0.1843727, upper bound: 0.1845280
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 230.16
Output dim: 4, lower bound: -0.1844099, upper bound: 0.1844886
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 230.16
Output dim: 4, lower bound: -0.1845284, upper bound: 0.1845381
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 230.16
Output dim: 4, lower bound: -0.1846316, upper bound: 0.1844304
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 230.16
Output dim: 4, lower bound: -0.1846143, upper bound: 0.1845333
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 230.16
Output dim: 4, lower bound: -0.1846168, upper bound: 0.1845304
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 230.16
Output dim: 4, lower bound: -0.1844332, upper bound: 0.1844393
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 230.16
Output dim: 4, lower bound: -0.1844429, upper bound: 0.1844335
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 230.16
Output dim: 4, lower bound: -0.1842659, upper bound: 0.1845072
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 230.16
Output dim: 4, lower bound: -0.1845146, upper bound: 0.1842683
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 230.16
Output dim: 4, lower bound: -0.1846298, upper bound: 0.1846338
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 230.16
Output dim: 4, lower bound: -0.1846259, upper bound: 0.1846307
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 230.16
Output dim: 4, lower bound: -0.1846400, upper bound: 0.1846453
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 230.16
Output dim: 4, lower bound: -0.1846400, upper bound: 0.1846395

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6928467, 0.6928301
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9253337, 2.9253650
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2475690, 0.2475841
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1974585, 1.1975108
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8761797, 0.8761916
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2872803, 1.2873149
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0487187, 1.0488877
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7214459, 1.7214704
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6084652, 2.6084666
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584443, 1.8584449

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2059

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2885

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1843986, upper bound: 0.1845225
time: 16.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844038, upper bound: 0.1845120
time: 68.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6928464, 0.6928303
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9253223, 2.9253764
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2475777, 0.2475754
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1974938, 1.1974758
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8761921, 0.8761795
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2873011, 1.2872944
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -1.0488255, 1.0487809
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.7214714, 1.7214451
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.6084652, 2.6084666
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8584446, 1.8584447

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2608

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844263, upper bound: 0.1844771
time: 12.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844240, upper bound: 0.1844691
time: 356.75 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 373.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 373.37
Output dim: 4, lower bound: -0.1843986, upper bound: 0.1845225
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 373.37
Output dim: 4, lower bound: -0.1844038, upper bound: 0.1845120
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 373.37
Output dim: 4, lower bound: -0.1844263, upper bound: 0.1844771
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 373.37
Output dim: 4, lower bound: -0.1844240, upper bound: 0.1844691
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 373.37
Output dim: 4, lower bound: -0.1843727, upper bound: 0.1845280
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 373.37
Output dim: 4, lower bound: -0.1844099, upper bound: 0.1844886
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 373.37
Output dim: 4, lower bound: -0.1845284, upper bound: 0.1845381
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 373.37
Output dim: 4, lower bound: -0.1846316, upper bound: 0.1844304
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 373.37
Output dim: 4, lower bound: -0.1846143, upper bound: 0.1845333
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 373.37
Output dim: 4, lower bound: -0.1846168, upper bound: 0.1845304
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 373.37
Output dim: 4, lower bound: -0.1844332, upper bound: 0.1844393
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 373.37
Output dim: 4, lower bound: -0.1844429, upper bound: 0.1844335
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 373.37
Output dim: 4, lower bound: -0.1842659, upper bound: 0.1845072
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 373.37
Output dim: 4, lower bound: -0.1845146, upper bound: 0.1842683
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 373.37
Output dim: 4, lower bound: -0.1846298, upper bound: 0.1846338
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 373.37
Output dim: 4, lower bound: -0.1846259, upper bound: 0.1846307
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 373.37
Output dim: 4, lower bound: -0.1846400, upper bound: 0.1846453
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 373.37
Output dim: 4, lower bound: -0.1846400, upper bound: 0.1846395
Binary search (step 0): status=Status.UNKNOWN, k_low=2, k_high=8, k_mid=5, eps_mid=0.0195312, abs_max=0.8765864372253418
rel_dist={4: [-0.18467027294666494, 0.18467521355340155]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2638

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427640, upper bound: 0.1427642
time: 210.16 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427594, upper bound: 0.1427673
time: 133.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 343.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 343.70
Output dim: 4, lower bound: -0.1427640, upper bound: 0.1427642
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 343.70
Output dim: 4, lower bound: -0.1427594, upper bound: 0.1427673

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814643, 0.6814643
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023397, 2.9023399
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348186, 0.2348186
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767589, 1.1767588
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554122, 0.8554124
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646852, 1.2646850
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825097, 0.9825096
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795120, 1.6795117
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946677, 2.5946679
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319979, 1.8319979

Time for backsubstitution: 4.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2197

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427025, upper bound: 0.1427613
time: 28.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427587, upper bound: 0.1427094
time: 196.37 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814643, 0.6814642
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023397, 2.9023399
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348186, 0.2348186
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767589, 1.1767588
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554122, 0.8554124
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646849, 1.2646850
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825097, 0.9825096
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795115, 1.6795118
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946677, 2.5946679
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319979, 1.8319979

Time for backsubstitution: 4.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2410

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2414

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427647, upper bound: 0.1427640
time: 91.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427647, upper bound: 0.1427679
time: 329.00 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 424.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 424.28
Output dim: 4, lower bound: -0.1427025, upper bound: 0.1427613
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 424.28
Output dim: 4, lower bound: -0.1427587, upper bound: 0.1427094
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 424.28
Output dim: 4, lower bound: -0.1427647, upper bound: 0.1427640
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 424.28
Output dim: 4, lower bound: -0.1427647, upper bound: 0.1427679

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814103, 0.6814101
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023404, 2.9023409
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2347851, 0.2347857
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767992, 1.1767988
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554387, 0.8554388
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646024, 1.2646047
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825082, 0.9825003
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6793168, 1.6793239
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5947285, 2.5947292
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8321869, 1.8321834

Time for backsubstitution: 4.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 870

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3086

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426414, upper bound: 0.1427654
time: 25.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427024, upper bound: 0.1426973
time: 159.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814101, 0.6814104
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023407, 2.9023404
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2347857, 0.2347851
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767987, 1.1767992
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554387, 0.8554387
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646046, 1.2646024
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825003, 0.9825082
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6793239, 1.6793168
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5947294, 2.5947280
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8321836, 1.8321869

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2350

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 801

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427228, upper bound: 0.1426796
time: 19.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427237, upper bound: 0.1426792
time: 13.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814643, 0.6814642
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023397, 2.9023399
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348186, 0.2348186
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767589, 1.1767588
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554122, 0.8554124
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646849, 1.2646850
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825097, 0.9825096
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795115, 1.6795118
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946677, 2.5946679
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319979, 1.8319979

Time for backsubstitution: 4.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2890

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 886

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427604, upper bound: 0.1427670
time: 258.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427604, upper bound: 0.1427634
time: 216.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814643, 0.6814642
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023397, 2.9023399
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348186, 0.2348186
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767589, 1.1767588
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554122, 0.8554124
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646849, 1.2646850
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825097, 0.9825096
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795115, 1.6795118
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946677, 2.5946679
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319979, 1.8319979

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2623

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426730, upper bound: 0.1427246
time: 185.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427210, upper bound: 0.1426854
time: 17.12 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 207.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 207.16
Output dim: 4, lower bound: -0.1426414, upper bound: 0.1427654
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 207.16
Output dim: 4, lower bound: -0.1427024, upper bound: 0.1426973
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 207.16
Output dim: 4, lower bound: -0.1427228, upper bound: 0.1426796
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 207.16
Output dim: 4, lower bound: -0.1427237, upper bound: 0.1426792
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 207.16
Output dim: 4, lower bound: -0.1427604, upper bound: 0.1427670
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 207.16
Output dim: 4, lower bound: -0.1427604, upper bound: 0.1427634
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 207.16
Output dim: 4, lower bound: -0.1426730, upper bound: 0.1427246
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 207.16
Output dim: 4, lower bound: -0.1427210, upper bound: 0.1426854

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814088, 0.6814086
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023361, 2.9023361
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2347822, 0.2347828
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767987, 1.1767983
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554385, 0.8554385
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646017, 1.2646040
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825073, 0.9824994
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6793151, 1.6793221
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5947208, 2.5947218
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8321820, 1.8321784

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2888

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 753

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426140, upper bound: 0.1427572
time: 149.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426383, upper bound: 0.1427405
time: 140.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814089, 0.6814085
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023359, 2.9023361
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2347823, 0.2347828
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767987, 1.1767983
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554385, 0.8554384
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646017, 1.2646039
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825073, 0.9824994
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6793151, 1.6793222
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5947208, 2.5947218
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8321820, 1.8321784

Time for backsubstitution: 4.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 354

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1424532, upper bound: 0.1426942
time: 144.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426999, upper bound: 0.1424527
time: 302.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6813961, 0.6813962
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9022155, 2.9022164
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2346780, 0.2346852
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1758691, 1.1759440
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8550985, 0.8550950
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2637135, 1.2637973
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9815190, 0.9815516
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6783874, 1.6783888
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5947194, 2.5947185
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8321822, 1.8321857

Time for backsubstitution: 4.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3087

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2441

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427082, upper bound: 0.1426743
time: 32.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427107, upper bound: 0.1426663
time: 41.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6813960, 0.6813964
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9022169, 2.9022150
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2346858, 0.2346774
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1759437, 1.1758695
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8550951, 0.8550985
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2637997, 1.2637112
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9815438, 0.9815269
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6783962, 1.6783800
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5947194, 2.5947182
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8321822, 1.8321857

Time for backsubstitution: 4.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2064

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2352

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427143, upper bound: 0.1426625
time: 271.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427151, upper bound: 0.1426621
time: 146.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814643, 0.6814642
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023397, 2.9023399
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348186, 0.2348186
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767589, 1.1767588
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554122, 0.8554124
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646849, 1.2646850
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825097, 0.9825096
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795115, 1.6795118
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946677, 2.5946679
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319979, 1.8319979

Time for backsubstitution: 4.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2371

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426540, upper bound: 0.1426594
time: 161.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426540, upper bound: 0.1426467
time: 274.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814643, 0.6814642
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9023397, 2.9023399
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2348186, 0.2348186
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1767589, 1.1767588
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8554122, 0.8554124
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2646849, 1.2646850
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9825097, 0.9825096
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6795115, 1.6795118
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946677, 2.5946679
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319979, 1.8319979

Time for backsubstitution: 4.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2542

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 728

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427471, upper bound: 0.1427703
time: 13.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427591, upper bound: 0.1427644
time: 16.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6814162, 0.6814160
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.9021153, 2.9020934
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2346786, 0.2346771
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1714350, 1.1719129
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8540082, 0.8540363
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2593656, 1.2598464
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9809788, 0.9810194
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6740963, 1.6746306
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5946610, 2.5946617
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8319919, 1.8319925

Time for backsubstitution: 4.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 581

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2816

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426247, upper bound: 0.1426725
time: 372.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426237, upper bound: 0.1426770
time: 18.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 395.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 395.39
Output dim: 4, lower bound: -0.1426140, upper bound: 0.1427572
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 395.39
Output dim: 4, lower bound: -0.1426383, upper bound: 0.1427405
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 395.39
Output dim: 4, lower bound: -0.1424532, upper bound: 0.1426942
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 395.39
Output dim: 4, lower bound: -0.1426999, upper bound: 0.1424527
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 395.39
Output dim: 4, lower bound: -0.1427082, upper bound: 0.1426743
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 395.39
Output dim: 4, lower bound: -0.1427107, upper bound: 0.1426663
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 395.39
Output dim: 4, lower bound: -0.1427143, upper bound: 0.1426625
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 395.39
Output dim: 4, lower bound: -0.1427151, upper bound: 0.1426621
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 395.39
Output dim: 4, lower bound: -0.1426540, upper bound: 0.1426594
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 395.39
Output dim: 4, lower bound: -0.1426540, upper bound: 0.1426467
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 395.39
Output dim: 4, lower bound: -0.1427471, upper bound: 0.1427703
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 395.39
Output dim: 4, lower bound: -0.1427591, upper bound: 0.1427644
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 395.39
Output dim: 4, lower bound: -0.1426247, upper bound: 0.1426725
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 395.39
Output dim: 4, lower bound: -0.1426237, upper bound: 0.1426770
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 395.39
Output dim: 4, lower bound: -0.1427210, upper bound: 0.1426854
Binary search (step 1): status=Status.UNKNOWN, k_low=2, k_high=4, k_mid=3, eps_mid=0.0117188, abs_max=0.8554123640060425
rel_dist={4: [-0.1427653512292979, 0.1427739341779526]}

## Binary search (step 2) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2088
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 3504

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2088

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216721, upper bound: 0.1216850
time: 19.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216674, upper bound: 0.1216835
time: 398.98 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 418.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 418.60
Output dim: 4, lower bound: -0.1216721, upper bound: 0.1216850
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 418.60
Output dim: 4, lower bound: -0.1216674, upper bound: 0.1216835

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6756301, 0.6756233
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8904462, 2.8904862
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283587, 0.2283584
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1641846, 1.1642370
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8447567, 0.8447503
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2512976, 1.2513508
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9480277, 0.9480356
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6569793, 1.6570116
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877953, 2.5877953
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8190287, 1.8190440

Time for backsubstitution: 4.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 561

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3308

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216254, upper bound: 0.1216292
time: 91.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216254, upper bound: 0.1216239
time: 61.95 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6756233, 0.6756301
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8904862, 2.8904462
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283584, 0.2283586
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1642370, 1.1641845
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8447503, 0.8447567
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2513508, 1.2512976
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9480356, 0.9480278
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6570117, 1.6569793
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877953, 2.5877953
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8190440, 1.8190285

Time for backsubstitution: 4.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2410

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 831

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216436, upper bound: 0.1216838
time: 413.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216628, upper bound: 0.1216724
time: 21.65 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 438.81 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 438.81
Output dim: 4, lower bound: -0.1216254, upper bound: 0.1216292
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 438.81
Output dim: 4, lower bound: -0.1216254, upper bound: 0.1216239
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 438.81
Output dim: 4, lower bound: -0.1216436, upper bound: 0.1216838
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 438.81
Output dim: 4, lower bound: -0.1216628, upper bound: 0.1216724

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6756300, 0.6756232
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8904467, 2.8904862
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283584, 0.2283605
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1641841, 1.1642425
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8447565, 0.8447508
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2512994, 1.2513498
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9480268, 0.9480565
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6569929, 1.6570041
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877953, 2.5877953
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8190285, 1.8190430

Time for backsubstitution: 4.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 607

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216167, upper bound: 0.1216316
time: 16.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216128, upper bound: 0.1216351
time: 11.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6756300, 0.6756233
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8904459, 2.8904862
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283587, 0.2283581
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1641846, 1.1642365
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8447567, 0.8447502
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2512968, 1.2513508
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9480277, 0.9480344
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6569719, 1.6570116
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877953, 2.5877953
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8190287, 1.8190438

Time for backsubstitution: 4.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2198

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2443

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216174, upper bound: 0.1216273
time: 266.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216169, upper bound: 0.1216356
time: 201.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6756229, 0.6756297
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8904855, 2.8904452
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283580, 0.2283583
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1642370, 1.1641843
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8447500, 0.8447564
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2513508, 1.2512976
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9480339, 0.9480263
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6570117, 1.6569793
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877953, 2.5877948
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8190439, 1.8190285

Time for backsubstitution: 4.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 890

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3308

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215964, upper bound: 0.1216385
time: 180.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215964, upper bound: 0.1216211
time: 321.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6756229, 0.6756297
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8904855, 2.8904452
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283580, 0.2283583
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1642370, 1.1641843
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8447500, 0.8447564
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2513508, 1.2512976
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9480339, 0.9480262
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6570117, 1.6569793
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877953, 2.5877953
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8190439, 1.8190285

Time for backsubstitution: 4.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2527

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 206

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216692, upper bound: 0.1216577
time: 228.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216591, upper bound: 0.1216605
time: 24.54 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 257.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 257.12
Output dim: 4, lower bound: -0.1216167, upper bound: 0.1216316
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 257.12
Output dim: 4, lower bound: -0.1216128, upper bound: 0.1216351
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 257.12
Output dim: 4, lower bound: -0.1216174, upper bound: 0.1216273
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 257.12
Output dim: 4, lower bound: -0.1216169, upper bound: 0.1216356
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 257.12
Output dim: 4, lower bound: -0.1215964, upper bound: 0.1216385
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 257.12
Output dim: 4, lower bound: -0.1215964, upper bound: 0.1216211
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 257.12
Output dim: 4, lower bound: -0.1216692, upper bound: 0.1216577
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 257.12
Output dim: 4, lower bound: -0.1216591, upper bound: 0.1216605

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6754822, 0.6754665
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8903706, 2.8904107
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283732, 0.2283798
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1636708, 1.1637440
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8447529, 0.8447519
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2507349, 1.2507973
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9467183, 0.9467955
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6569786, 1.6569889
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5878901, 2.5878897
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8190130, 1.8190291

Time for backsubstitution: 4.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2236

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3062

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1215306, upper bound: 0.1215243
time: 171.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1215306, upper bound: 0.1215286
time: 181.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6754733, 0.6754753
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8903713, 2.8904102
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283777, 0.2283753
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1636856, 1.1637293
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8447574, 0.8447472
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2507470, 1.2507852
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9467657, 0.9467483
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6569779, 1.6569896
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5878897, 2.5878897
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8190147, 1.8190272

Time for backsubstitution: 4.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 3148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2183

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215855, upper bound: 0.1216182
time: 28.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216055, upper bound: 0.1215949
time: 153.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6756334, 0.6756265
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8904450, 2.8904850
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283608, 0.2283604
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1641848, 1.1642370
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8447565, 0.8447499
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2512970, 1.2513512
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9480277, 0.9480344
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6569803, 1.6570199
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877934, 2.5877929
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8190277, 1.8190433

Time for backsubstitution: 4.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 742

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215986, upper bound: 0.1215744
time: 21.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215600, upper bound: 0.1216168
time: 13.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6756334, 0.6756265
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8904448, 2.8904850
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283609, 0.2283603
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1641848, 1.1642370
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8447565, 0.8447499
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2512970, 1.2513514
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9480277, 0.9480343
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6569803, 1.6570200
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877929, 2.5877929
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8190277, 1.8190433

Time for backsubstitution: 4.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2646

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215495, upper bound: 0.1215703
time: 167.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215630, upper bound: 0.1215519
time: 483.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6756228, 0.6756297
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8904858, 2.8904452
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283577, 0.2283604
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1642368, 1.1641899
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8447498, 0.8447570
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2513525, 1.2512965
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9480327, 0.9480472
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6570253, 1.6569717
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877953, 2.5877950
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8190436, 1.8190273

Time for backsubstitution: 4.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2893

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215938, upper bound: 0.1216245
time: 16.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216199, upper bound: 0.1216363
time: 31.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6756228, 0.6756297
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8904853, 2.8904452
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283580, 0.2283581
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1642370, 1.1641841
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8447500, 0.8447562
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2513498, 1.2512976
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9480339, 0.9480250
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6570041, 1.6569793
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877953, 2.5877950
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8190439, 1.8190283

Time for backsubstitution: 4.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 206
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2542
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2191

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2346

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1214442, upper bound: 0.1214610
time: 54.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1214274, upper bound: 0.1214873
time: 19.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 0.9345794, 1.8825785, 0.9345794, 1.8825785, -0.6756203, 0.6756271
1: -3.3633165, -0.0221298, -3.3633165, -0.0221298, -2.8904848, 2.8904448
2: 0.3617328, 1.0402305, 0.3617328, 1.0402305, -0.2283569, 0.2283572
3: -2.1392312, -0.3277394, -2.1392312, -0.3277394, -1.1642365, 1.1641841
4: -2.0234284, -0.5932747, -2.0234284, -0.5932747, -0.8447496, 0.8447559
5: -2.2512624, -0.3884894, -2.2512624, -0.3884894, -1.2513497, 1.2512965
6: -6.3656788, -3.0180762, -6.3656788, -3.0180762, -0.9480314, 0.9480234
7: -2.6508508, 0.3194579, -2.6508508, 0.3194579, -1.6570091, 1.6569767
8: -2.6750135, 0.1594995, -2.6750135, 0.1594995, -2.5877934, 2.5877931
9: -3.6092696, -1.2809148, -3.6092696, -1.2809148, -1.8190434, 1.8190281

Time for backsubstitution: 4.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2330
type: RSZ, layer: 1, pos: 2238
type: RSZ, layer: 1, pos: 2528
type: RSZ, layer: 1, pos: 2578
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 2575
type: RSZ, layer: 1, pos: 3216
type: RSZ, layer: 1, pos: 2652
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 2963
type: RSZ, layer: 1, pos: 2614
type: RSZ, layer: 1, pos: 3148
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2801
type: RSZ, layer: 1, pos: 2906
type: RSZ, layer: 1, pos: 2164
type: RSZ, layer: 1, pos: 2239
type: RSZ, layer: 1, pos: 2894
type: RSZ, layer: 1, pos: 2885
type: RSZ, layer: 1, pos: 2440
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 2665
type: RSZ, layer: 1, pos: 2167
type: RSZ, layer: 1, pos: 2903
type: RSZ, layer: 1, pos: 2409
type: RSZ, layer: 1, pos: 2237
type: RSZ, layer: 1, pos: 2207
type: RSZ, layer: 1, pos: 2653
type: RSZ, layer: 1, pos: 2202
type: RSZ, layer: 1, pos: 2923
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2816
type: RSZ, layer: 1, pos: 2526
type: RSZ, layer: 1, pos: 2064
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2156
type: RSZ, layer: 1, pos: 2130
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 290
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 2096
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 2063
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 2196
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2638
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2414
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2039
type: RSZ, layer: 1, pos: 2887
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 2513
type: RSZ, layer: 1, pos: 3086
type: RSZ, layer: 1, pos: 3518
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 3438
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 3080
type: RSZ, layer: 1, pos: 2373
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2047
type: RSZ, layer: 1, pos: 2148
type: RSZ, layer: 1, pos: 2183
type: RSZ, layer: 1, pos: 2197
type: RSZ, layer: 1, pos: 354
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2201
type: RSZ, layer: 1, pos: 2236
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 2389
type: RSZ, layer: 1, pos: 2886
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2883
type: RSZ, layer: 1, pos: 2386
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 2168
type: RSZ, layer: 1, pos: 2048
type: RSZ, layer: 1, pos: 2904
type: RSZ, layer: 1, pos: 2442
type: RSZ, layer: 1, pos: 2390
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 2439
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2613
type: RSZ, layer: 1, pos: 2922
type: RSZ, layer: 1, pos: 2182
type: RSZ, layer: 1, pos: 2145
type: RSZ, layer: 1, pos: 3076
type: RSZ, layer: 1, pos: 2893
type: RSZ, layer: 1, pos: 2387
type: RSZ, layer: 1, pos: 888
type: RSZ, layer: 1, pos: 2166
type: RSZ, layer: 1, pos: 2059
type: RSZ, layer: 1, pos: 2634
type: RSZ, layer: 1, pos: 3279
type: RSZ, layer: 1, pos: 2962
type: RSZ, layer: 1, pos: 2890
type: RSZ, layer: 1, pos: 2193
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 2227
type: RSZ, layer: 1, pos: 2288
type: RSZ, layer: 1, pos: 2350
type: RSZ, layer: 1, pos: 434
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 3075
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 2628
type: RSZ, layer: 1, pos: 2608
type: RSZ, layer: 1, pos: 2298
type: RSZ, layer: 1, pos: 2646
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 2889
type: RSZ, layer: 1, pos: 3062
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2285
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 887
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 2529
type: RSZ, layer: 1, pos: 2220
type: RSZ, layer: 1, pos: 2884
type: RSZ, layer: 1, pos: 3149
type: RSZ, layer: 1, pos: 152
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 2892
type: RSZ, layer: 1, pos: 2661
type: RSZ, layer: 1, pos: 2191
type: RSZ, layer: 1, pos: 2319
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2049
type: RSZ, layer: 1, pos: 2165
type: RSZ, layer: 1, pos: 2867
type: RSZ, layer: 1, pos: 2094
type: RSZ, layer: 1, pos: 3011
type: RSZ, layer: 1, pos: 2314
type: RSZ, layer: 1, pos: 3504
type: RSZ, layer: 1, pos: 2185
type: RSZ, layer: 1, pos: 2500
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2593
type: RSZ, layer: 1, pos: 2171
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2479
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 2194
type: RSZ, layer: 1, pos: 2861
type: RSZ, layer: 1, pos: 2990
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 3308
type: RSZ, layer: 1, pos: 3109
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 2050
type: RSZ, layer: 1, pos: 3007
type: RSZ, layer: 1, pos: 2527
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2346
type: RSZ, layer: 1, pos: 2444
type: RSZ, layer: 1, pos: 2891
type: RSZ, layer: 1, pos: 2413
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 3131
type: RSZ, layer: 1, pos: 3132
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 500
type: RSZ, layer: 1, pos: 2882
type: RSZ, layer: 1, pos: 2388
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 2866
type: RSZ, layer: 1, pos: 141
type: RSZ, layer: 1, pos: 3516
type: RSZ, layer: 1, pos: 3079
type: RSZ, layer: 1, pos: 2623
type: RSZ, layer: 1, pos: 3069
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2457
type: RSZ, layer: 1, pos: 2060
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2888
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2095
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 2192
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2228
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2352
type: RSZ, layer: 1, pos: 142
type: RSZ, layer: 1, pos: 2371
type: RSZ, layer: 1, pos: 2320
type: RSZ, layer: 1, pos: 3068
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2630
type: RSZ, layer: 1, pos: 2443
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 371
type: RSZ, layer: 1, pos: 2195
type: RSZ, layer: 1, pos: 2385
type: RSZ, layer: 1, pos: 2542

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 143

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1214833, upper bound: 0.1216501
time: 170.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216535, upper bound: 0.1214819
time: 153.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 328.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 328.53
Output dim: 4, lower bound: -0.1215306, upper bound: 0.1215243
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 328.53
Output dim: 4, lower bound: -0.1215306, upper bound: 0.1215286
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 328.53
Output dim: 4, lower bound: -0.1215855, upper bound: 0.1216182
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 328.53
Output dim: 4, lower bound: -0.1216055, upper bound: 0.1215949
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 328.53
Output dim: 4, lower bound: -0.1215986, upper bound: 0.1215744
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 328.53
Output dim: 4, lower bound: -0.1215600, upper bound: 0.1216168
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 328.53
Output dim: 4, lower bound: -0.1215495, upper bound: 0.1215703
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 328.53
Output dim: 4, lower bound: -0.1215630, upper bound: 0.1215519
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 328.53
Output dim: 4, lower bound: -0.1215938, upper bound: 0.1216245
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 328.53
Output dim: 4, lower bound: -0.1216199, upper bound: 0.1216363
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 328.53
Output dim: 4, lower bound: -0.1214442, upper bound: 0.1214610
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 328.53
Output dim: 4, lower bound: -0.1214274, upper bound: 0.1214873
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 328.53
Output dim: 4, lower bound: -0.1214833, upper bound: 0.1216501
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 328.53
Output dim: 4, lower bound: -0.1216535, upper bound: 0.1214819
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 328.53
Output dim: 4, lower bound: -0.1216591, upper bound: 0.1216605
Binary search (step 2): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=0.8448255062103271
rel_dist={4: [-0.1216752063496438, 0.12168907463758383]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 12634.63 seconds
