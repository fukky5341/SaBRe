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
execution time: IAR + LP analysis = 5.69 + 19.99 = 25.68 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 17974.32 seconds, max iter: 100)

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
Binary search time: 286.08 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Individual Split (IS_dual_ind) starts
Time budget: 17688.24 seconds

## Binary search (step 0) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 290
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 290

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846686, upper bound: 0.1838047
time: 20.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1846677, upper bound: 0.1846707
time: 21.19 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 41.66 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 41.66
Output dim: 4, lower bound: -0.1846686, upper bound: 0.1838047
IS_A2, status: Status.UNKNOWN, split count: 1, time: 41.66
Output dim: 4, lower bound: -0.1846677, upper bound: 0.1846707

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9373658, 1.8808306, 0.9348752, 1.8811090, -0.6889743, 0.6910908
1: -3.3615472, -0.0222447, -3.3619015, -0.0232708, -2.9223971, 2.9239702
2: 0.3615273, 1.0395275, 0.3618716, 1.0396456, -0.2454911, 0.2460525
3: -2.1354001, -0.3302180, -2.1363173, -0.3280894, -1.1940105, 1.1928041
4: -2.0165973, -0.6043231, -2.0228701, -0.6016386, -0.8613904, 0.8650042
5: -2.2459154, -0.3914326, -2.2472184, -0.3890206, -1.2822009, 1.2811216
6: -6.3506284, -3.0248573, -6.3541298, -3.0181241, -1.0353919, 1.0321440
7: -2.6405604, 0.3148721, -2.6438799, 0.3194184, -1.7130272, 1.7105395
8: -2.6678741, 0.1454756, -2.6742544, 0.1490826, -2.5899296, 2.5940182
9: -3.6063704, -1.2813771, -3.6071148, -1.2812343, -1.8555864, 1.8561054

Time for backsubstitution: 4.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 3068

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845896, upper bound: 0.1835989
time: 147.51 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845879, upper bound: 0.1837226
time: 139.82 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9346080, 1.8825622, 0.9346012, 1.8825667, -0.6928104, 0.6909976
1: -3.3612962, -0.0221319, -3.3618042, -0.0221314, -2.9236863, 2.9241397
2: 0.3617358, 1.0399849, 0.3617349, 1.0400511, -0.2477214, 0.2458035
3: -2.1392198, -0.3277416, -2.1392229, -0.3277410, -1.1968616, 1.1981452
4: -2.0234258, -0.5932908, -2.0234265, -0.5932868, -0.8765788, 0.8689389
5: -2.2512388, -0.3884903, -2.2512455, -0.3884900, -1.2855437, 1.2879159
6: -6.3655934, -3.0180769, -6.3656163, -3.0180769, -1.0302069, 1.0503485
7: -2.6508150, 0.3194544, -2.6508245, 0.3194556, -1.7121329, 1.7221537
8: -2.6749899, 0.1594675, -2.6749949, 0.1594758, -2.6096556, 2.6080875
9: -3.6082540, -1.2810316, -3.6084776, -1.2810001, -1.8581507, 1.8570685

Time for backsubstitution: 4.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 3068

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845900, upper bound: 0.1844670
time: 15.23 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845898, upper bound: 0.1845931
time: 19.27 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 38.67 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 38.67
Output dim: 4, lower bound: -0.1845896, upper bound: 0.1835989
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 38.67
Output dim: 4, lower bound: -0.1845879, upper bound: 0.1837226
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 38.67
Output dim: 4, lower bound: -0.1845900, upper bound: 0.1844670
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 38.67
Output dim: 4, lower bound: -0.1845898, upper bound: 0.1845931

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.9374033, 1.8805383, 0.9349355, 1.8807318, -0.6885748, 0.6907351
1: -3.3611062, -0.0285993, -3.3593452, -0.0310616, -2.9144044, 2.9155030
2: 0.3622912, 1.0395267, 0.3628137, 1.0394582, -0.2446062, 0.2451673
3: -2.1295519, -0.3302196, -2.1293139, -0.3292894, -1.1867249, 1.1852243
4: -2.0137782, -0.6043340, -2.0193610, -0.6015322, -0.8580893, 0.8614100
5: -2.2396355, -0.3914394, -2.2395194, -0.3908566, -1.2742407, 1.2730641
6: -6.3452516, -3.0248594, -6.3473883, -3.0182345, -1.0282304, 1.0248494
7: -2.6355603, 0.3148627, -2.6375227, 0.3172684, -1.7060606, 1.7041991
8: -2.6677291, 0.1427475, -2.6733804, 0.1456739, -2.5864646, 2.5905375
9: -3.6061151, -1.2828773, -3.6061454, -1.2831188, -1.8531702, 1.8529117

Time for backsubstitution: 4.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1843526, upper bound: 0.1834452
time: 226.68 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845092, upper bound: 0.1834526
time: 15.21 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.9373780, 1.8806798, 0.9348894, 1.8809227, -0.6887667, 0.6909599
1: -3.3614469, -0.0232894, -3.3617892, -0.0243492, -2.9199228, 2.9228499
2: 0.3616186, 1.0395273, 0.3619742, 1.0396452, -0.2454272, 0.2452691
3: -2.1341720, -0.3302190, -2.1348305, -0.3280905, -1.1934562, 1.1862533
4: -2.0153594, -0.6043251, -2.0213089, -0.6016411, -0.8605337, 0.8618542
5: -2.2447000, -0.3914355, -2.2457480, -0.3890239, -1.2816944, 1.2741843
6: -6.3494687, -3.0248585, -6.3527346, -3.0181260, -1.0344985, 1.0260167
7: -2.6394973, 0.3148695, -2.6426105, 0.3194149, -1.7127728, 1.7034438
8: -2.6678264, 0.1445804, -2.6741958, 0.1480341, -2.5886860, 2.5930347
9: -3.6062934, -1.2820222, -3.6070242, -1.2820551, -1.8548894, 1.8555505

Time for backsubstitution: 4.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1843519, upper bound: 0.1836494
time: 18.30 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845093, upper bound: 0.1836474
time: 13.41 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.9346452, 1.8822699, 0.9346619, 1.8821893, -0.6924113, 0.6906421
1: -3.3608422, -0.0284896, -3.3592379, -0.0299220, -2.9156871, 2.9156671
2: 0.3624991, 1.0399843, 0.3626771, 1.0398638, -0.2468366, 0.2449180
3: -2.1333706, -0.3277429, -2.1322196, -0.3289406, -1.1895750, 1.1905648
4: -2.0206010, -0.5933022, -2.0199161, -0.5931804, -0.8732715, 0.8653389
5: -2.2449584, -0.3884972, -2.2435455, -0.3903269, -1.2775829, 1.2798584
6: -6.3602171, -3.0180793, -6.3588753, -3.0181868, -1.0230455, 1.0430541
7: -2.6458154, 0.3194449, -2.6444666, 0.3173052, -1.7051650, 1.7158096
8: -2.6748445, 0.1567397, -2.6741207, 0.1560680, -2.6061881, 2.6046052
9: -3.6079969, -1.2825327, -3.6075058, -1.2828846, -1.8557324, 1.8538730

Time for backsubstitution: 4.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1843509, upper bound: 0.1843179
time: 13.23 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845081, upper bound: 0.1843172
time: 16.30 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.9346198, 1.8824114, 0.9346157, 1.8823806, -0.6926032, 0.6908669
1: -3.3611929, -0.0231752, -3.3616908, -0.0232074, -2.9212127, 2.9230196
2: 0.3618267, 1.0399849, 0.3618374, 1.0400511, -0.2476579, 0.2450197
3: -2.1379910, -0.3277427, -2.1377363, -0.3277422, -1.1963077, 1.1915923
4: -2.0221858, -0.5932928, -2.0218647, -0.5932891, -0.8757201, 0.8657854
5: -2.2500231, -0.3884933, -2.2497752, -0.3884942, -1.2850375, 1.2809771
6: -6.3644361, -3.0180774, -6.3642225, -3.0180781, -1.0293136, 1.0442212
7: -2.6497524, 0.3194519, -2.6495554, 0.3194523, -1.7118790, 1.7150545
8: -2.6749415, 0.1585711, -2.6749375, 0.1584273, -2.6084101, 2.6071029
9: -3.6081769, -1.2816772, -3.6083863, -1.2818215, -1.8574535, 1.8565135

Time for backsubstitution: 4.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1843512, upper bound: 0.1845115
time: 16.21 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1845090, upper bound: 0.1845164
time: 240.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 260.89 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 260.89
Output dim: 4, lower bound: -0.1843526, upper bound: 0.1834452
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 260.89
Output dim: 4, lower bound: -0.1845092, upper bound: 0.1834526
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 260.89
Output dim: 4, lower bound: -0.1843519, upper bound: 0.1836494
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 260.89
Output dim: 4, lower bound: -0.1845093, upper bound: 0.1836474
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 260.89
Output dim: 4, lower bound: -0.1843509, upper bound: 0.1843179
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 260.89
Output dim: 4, lower bound: -0.1845081, upper bound: 0.1843172
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 260.89
Output dim: 4, lower bound: -0.1843512, upper bound: 0.1845115
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 260.89
Output dim: 4, lower bound: -0.1845090, upper bound: 0.1845164

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9374410, 1.8801787, 0.9349648, 1.8804545, -0.6882645, 0.6903498
1: -3.3607020, -0.0341954, -3.3590305, -0.0353851, -2.9097795, 2.9096696
2: 0.3628883, 1.0395262, 0.3632742, 1.0394577, -0.2439207, 0.2446250
3: -2.1234925, -0.3302206, -2.1246376, -0.3292903, -1.1808319, 1.1806750
4: -2.0111067, -0.6043449, -2.0172937, -0.6015409, -0.8554683, 0.8593770
5: -2.2337103, -0.3914458, -2.2349453, -0.3908614, -1.2680866, 1.2683023
6: -6.3405285, -3.0248604, -6.3436069, -3.0182357, -1.0227584, 1.0204902
7: -2.6307445, 0.3148531, -2.6338062, 0.3172610, -1.7012974, 1.7005136
8: -2.6675823, 0.1399624, -2.6732674, 0.1435255, -2.5842228, 2.5876727
9: -3.6058643, -1.2845739, -3.6059513, -1.2844276, -1.8518027, 1.8511738

Time for backsubstitution: 4.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1842765, upper bound: 0.1830666
time: 30.57 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1842863, upper bound: 0.1833838
time: 171.27 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9372791, 1.8801720, 0.9349664, 1.8804430, -0.6884483, 0.6903297
1: -3.3687172, -0.0331669, -3.3590415, -0.0353205, -2.9185739, 2.9105902
2: 0.3627425, 1.0399137, 0.3632680, 1.0394577, -0.2439969, 0.2455874
3: -2.1237240, -0.3265365, -2.1245849, -0.3292902, -1.1811814, 1.1893122
4: -2.0114613, -0.6029190, -2.0174365, -0.6015403, -0.8558748, 0.8626703
5: -2.2339025, -0.3879306, -2.2348003, -0.3908614, -1.2684807, 1.2766937
6: -6.3417640, -3.0217614, -6.3444571, -3.0182352, -1.0230658, 1.0276680
7: -2.6306973, 0.3187498, -2.6336725, 0.3172615, -1.7015958, 1.7090529
8: -2.6690569, 0.1398416, -2.6732593, 0.1433778, -2.5855334, 2.5875893
9: -3.6083536, -1.2845662, -3.6059384, -1.2844567, -1.8539566, 1.8510826

Time for backsubstitution: 4.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844323, upper bound: 0.1830736
time: 89.23 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844451, upper bound: 0.1833876
time: 11.36 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9374163, 1.8803191, 0.9349204, 1.8806448, -0.6884449, 0.6905586
1: -3.3609960, -0.0299416, -3.3614209, -0.0297391, -2.9142072, 2.9158003
2: 0.3623224, 1.0395266, 0.3625364, 1.0396447, -0.2445382, 0.2446652
3: -2.1274488, -0.3302200, -2.1294463, -0.3280914, -1.1858737, 1.1815372
4: -2.0122783, -0.6043375, -2.0189064, -0.6016505, -0.8571932, 0.8596611
5: -2.2381446, -0.3914420, -2.2404776, -0.3890289, -1.2738259, 1.2692009
6: -6.3439841, -3.0248604, -6.3483052, -3.0181274, -1.0274549, 1.0212623
7: -2.6342535, 0.3148597, -2.6383355, 0.3194070, -1.7064334, 1.6997571
8: -2.6676788, 0.1417488, -2.6740780, 0.1457329, -2.5863721, 2.5901620
9: -3.6060357, -1.2839143, -3.6068134, -1.2835439, -1.8534255, 1.8537700

Time for backsubstitution: 4.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1842732, upper bound: 0.1832660
time: 284.24 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1842853, upper bound: 0.1835692
time: 26.21 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9372431, 1.8803911, 0.9349117, 1.8807011, -0.6887063, 0.6906478
1: -3.3692062, -0.0255775, -3.3616157, -0.0265172, -2.9261656, 2.9201345
2: 0.3618229, 1.0399144, 0.3622019, 1.0396452, -0.2448508, 0.2456234
3: -2.1307700, -0.3265360, -2.1321044, -0.3280912, -1.1878104, 1.1895549
4: -2.0140388, -0.6029061, -2.0202370, -0.6016449, -0.8584106, 0.8628618
5: -2.2413430, -0.3879257, -2.2429848, -0.3890281, -1.2759752, 1.2771740
6: -6.3482194, -3.0217600, -6.3516808, -3.0181270, -1.0296130, 1.0287349
7: -2.6364858, 0.3187603, -2.6402626, 0.3194109, -1.7075100, 1.7066405
8: -2.6691930, 0.1424626, -2.6741061, 0.1463407, -2.5883079, 2.5908110
9: -3.6086082, -1.2831442, -3.6068783, -1.2829342, -1.8560535, 1.8542154

Time for backsubstitution: 4.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844305, upper bound: 0.1832699
time: 96.84 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844448, upper bound: 0.1832675
time: 140.95 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9346827, 1.8819110, 0.9346906, 1.8819122, -0.6921015, 0.6902570
1: -3.3604271, -0.0340929, -3.3589165, -0.0342488, -2.9110575, 2.9098291
2: 0.3630959, 1.0399837, 0.3631374, 1.0398633, -0.2461513, 0.2443758
3: -2.1273117, -0.3277442, -2.1275432, -0.3289416, -1.1836821, 1.1860160
4: -2.0179238, -0.5933130, -2.0178468, -0.5931889, -0.8706417, 0.8633044
5: -2.2390335, -0.3885034, -2.2389724, -0.3903316, -1.2714286, 1.2750965
6: -6.3554940, -3.0180802, -6.3550944, -3.0181875, -1.0175738, 1.0386955
7: -2.6409988, 0.3194352, -2.6407499, 0.3172981, -1.7003999, 1.7121239
8: -2.6746988, 0.1539539, -2.6740086, 0.1539181, -2.6039467, 2.6017416
9: -3.6077428, -1.2842290, -3.6073103, -1.2841933, -1.8543634, 1.8521343

Time for backsubstitution: 4.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1842754, upper bound: 0.1839395
time: 198.58 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1842853, upper bound: 0.1842562
time: 9.19 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9345206, 1.8819039, 0.9346924, 1.8819004, -0.6922857, 0.6902367
1: -3.3684421, -0.0330577, -3.3589287, -0.0341804, -2.9198518, 2.9107502
2: 0.3629498, 1.0403712, 0.3631312, 1.0398635, -0.2462277, 0.2453380
3: -2.1275437, -0.3240603, -2.1274905, -0.3289416, -1.1840315, 1.1946533
4: -2.0182793, -0.5918877, -2.0179899, -0.5931883, -0.8710500, 0.8665974
5: -2.2392251, -0.3849880, -2.2388265, -0.3903317, -1.2718226, 1.2834882
6: -6.3567314, -3.0149801, -6.3559437, -3.0181875, -1.0178810, 1.0458729
7: -2.6409521, 0.3233323, -2.6406159, 0.3172987, -1.7007002, 1.7206626
8: -2.6761737, 0.1538326, -2.6740010, 0.1537712, -2.6052582, 2.6016576
9: -3.6102321, -1.2842212, -3.6072969, -1.2842230, -1.8565176, 1.8520433

Time for backsubstitution: 4.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844311, upper bound: 0.1839340
time: 262.86 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844452, upper bound: 0.1842455
time: 312.83 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9346581, 1.8820510, 0.9346464, 1.8821023, -0.6922815, 0.6904657
1: -3.3607292, -0.0298345, -3.3613143, -0.0285985, -2.9154904, 2.9159670
2: 0.3625301, 1.0399841, 0.3623994, 1.0400506, -0.2467692, 0.2444160
3: -2.1312687, -0.3277435, -2.1323524, -0.3277429, -1.1887252, 1.1868759
4: -2.0190969, -0.5933053, -2.0194602, -0.5932991, -0.8723696, 0.8635903
5: -2.2434673, -0.3884999, -2.2445045, -0.3884989, -1.2771683, 1.2759939
6: -6.3589516, -3.0180793, -6.3597922, -3.0180788, -1.0222703, 1.0394673
7: -2.6445081, 0.3194416, -2.6452799, 0.3194443, -1.7055380, 1.7113676
8: -2.6747954, 0.1557408, -2.6748209, 0.1561264, -2.6060963, 2.6042309
9: -3.6079154, -1.2835697, -3.6081738, -1.2833099, -1.8559878, 1.8547320

Time for backsubstitution: 4.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1842760, upper bound: 0.1841353
time: 171.15 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1842873, upper bound: 0.1844535
time: 19.46 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9344850, 1.8821234, 0.9346380, 1.8821586, -0.6925429, 0.6905550
1: -3.3689461, -0.0254643, -3.3615150, -0.0253754, -2.9274528, 2.9203038
2: 0.3620307, 1.0403720, 0.3620652, 1.0400510, -0.2470816, 0.2453740
3: -2.1345890, -0.3240596, -2.1350098, -0.3277427, -1.1906617, 1.1948941
4: -2.0208631, -0.5918740, -2.0207918, -0.5932930, -0.8735948, 0.8667917
5: -2.2466657, -0.3849829, -2.2470112, -0.3884978, -1.2793171, 1.2839667
6: -6.3631868, -3.0149803, -6.3631678, -3.0180798, -1.0244282, 1.0469398
7: -2.6467402, 0.3233421, -2.6472068, 0.3194483, -1.7066163, 1.7182510
8: -2.6763096, 0.1564535, -2.6748478, 0.1567343, -2.6080332, 2.6048794
9: -3.6104891, -1.2827994, -3.6082387, -1.2827005, -1.8586161, 1.8551776

Time for backsubstitution: 4.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844334, upper bound: 0.1841278
time: 411.65 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844448, upper bound: 0.1844447
time: 152.50 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 568.52 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 568.52
Output dim: 4, lower bound: -0.1842765, upper bound: 0.1830666
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 568.52
Output dim: 4, lower bound: -0.1842863, upper bound: 0.1833838
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 568.52
Output dim: 4, lower bound: -0.1844323, upper bound: 0.1830736
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 568.52
Output dim: 4, lower bound: -0.1844451, upper bound: 0.1833876
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 568.52
Output dim: 4, lower bound: -0.1842732, upper bound: 0.1832660
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 568.52
Output dim: 4, lower bound: -0.1842853, upper bound: 0.1835692
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 568.52
Output dim: 4, lower bound: -0.1844305, upper bound: 0.1832699
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 568.52
Output dim: 4, lower bound: -0.1844448, upper bound: 0.1832675
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 568.52
Output dim: 4, lower bound: -0.1842754, upper bound: 0.1839395
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 568.52
Output dim: 4, lower bound: -0.1842853, upper bound: 0.1842562
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 568.52
Output dim: 4, lower bound: -0.1844311, upper bound: 0.1839340
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 568.52
Output dim: 4, lower bound: -0.1844452, upper bound: 0.1842455
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 568.52
Output dim: 4, lower bound: -0.1842760, upper bound: 0.1841353
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 568.52
Output dim: 4, lower bound: -0.1842873, upper bound: 0.1844535
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 568.52
Output dim: 4, lower bound: -0.1844334, upper bound: 0.1841278
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 568.52
Output dim: 4, lower bound: -0.1844448, upper bound: 0.1844447

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9375601, 1.8791304, 0.9355603, 1.8790936, -0.6867597, 0.6886964
1: -3.3596914, -0.0419812, -3.3516548, -0.0455289, -2.8988872, 2.8946695
2: 0.3632354, 1.0391420, 0.3637546, 1.0389496, -0.2430277, 0.2436626
3: -2.1116996, -0.3302311, -2.1092651, -0.3332289, -1.1623104, 1.1639063
4: -2.0079484, -0.6043628, -2.0131540, -0.6025226, -0.8508599, 0.8549689
5: -2.2214806, -0.3914878, -2.2189105, -0.3951438, -1.2486652, 1.2502440
6: -6.3342619, -3.0248616, -6.3354053, -3.0184364, -1.0163389, 1.0125766
7: -2.6190474, 0.3148347, -2.6186650, 0.3133012, -1.6792167, 1.6821278
8: -2.6670985, 0.1313720, -2.6695514, 0.1324213, -2.5732698, 2.5757437
9: -3.6049023, -1.2899323, -3.5982513, -1.2913692, -1.8452593, 1.8395574

Time for backsubstitution: 4.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2390

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1839284, upper bound: 0.1830179
time: 26.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1842678, upper bound: 0.1830177
time: 295.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9374569, 1.8800341, 0.9349839, 1.8802760, -0.6879924, 0.6901842
1: -3.3605716, -0.0353856, -3.3588724, -0.0368226, -2.9079616, 2.9084282
2: 0.3629333, 1.0395068, 0.3633279, 1.0394320, -0.2438311, 0.2445417
3: -2.1218147, -0.3302219, -2.1225920, -0.3292916, -1.1802549, 1.1681345
4: -2.0106268, -0.6043475, -2.0167129, -0.6015436, -0.8552395, 0.8559723
5: -2.2317863, -0.3914503, -2.2325866, -0.3908668, -1.2672888, 1.2556825
6: -6.3399272, -3.0248606, -6.3428698, -3.0182354, -1.0220358, 1.0187802
7: -2.6288397, 0.3148509, -2.6315210, 0.3172581, -1.7011554, 1.6843228
8: -2.6675143, 0.1387451, -2.6731880, 0.1420652, -2.5825033, 2.5862584
9: -3.6057439, -1.2851766, -3.6058059, -1.2850977, -1.8507831, 1.8503246

Time for backsubstitution: 4.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2390

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1839370, upper bound: 0.1833697
time: 14.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1842808, upper bound: 0.1833743
time: 46.45 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9373987, 1.8791233, 0.9355623, 1.8790818, -0.6869435, 0.6886763
1: -3.3677077, -0.0409532, -3.3516665, -0.0454650, -2.9076836, 2.8955917
2: 0.3630900, 1.0395294, 0.3637485, 1.0389496, -0.2431037, 0.2446252
3: -2.1119311, -0.3265471, -2.1092126, -0.3332287, -1.1626607, 1.1725448
4: -2.0083032, -0.6029373, -2.0132964, -0.6025223, -0.8512655, 0.8582637
5: -2.2216718, -0.3879729, -2.2187653, -0.3951437, -1.2490644, 1.2586374
6: -6.3354964, -3.0217621, -6.3362536, -3.0184364, -1.0166466, 1.0197563
7: -2.6190012, 0.3187313, -2.6185305, 0.3133013, -1.6795156, 1.6906670
8: -2.6685729, 0.1312509, -2.6695437, 0.1322735, -2.5745819, 2.5756600
9: -3.6073928, -1.2899244, -3.5982380, -1.2913986, -1.8474143, 1.8394661

Time for backsubstitution: 4.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2390

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1840879, upper bound: 0.1830140
time: 232.15 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1844254, upper bound: 0.1830246
time: 57.29 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 293.92 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 293.92
Output dim: 4, lower bound: -0.1839284, upper bound: 0.1830179
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 293.92
Output dim: 4, lower bound: -0.1842678, upper bound: 0.1830177
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 293.92
Output dim: 4, lower bound: -0.1839370, upper bound: 0.1833697
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 293.92
Output dim: 4, lower bound: -0.1842808, upper bound: 0.1833743
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 293.92
Output dim: 4, lower bound: -0.1840879, upper bound: 0.1830140
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 293.92
Output dim: 4, lower bound: -0.1844254, upper bound: 0.1830246
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 293.92
Output dim: 4, lower bound: -0.1844451, upper bound: 0.1833876
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 293.92
Output dim: 4, lower bound: -0.1842732, upper bound: 0.1832660
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 293.92
Output dim: 4, lower bound: -0.1842853, upper bound: 0.1835692
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 293.92
Output dim: 4, lower bound: -0.1844305, upper bound: 0.1832699
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 293.92
Output dim: 4, lower bound: -0.1844448, upper bound: 0.1832675
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 293.92
Output dim: 4, lower bound: -0.1842754, upper bound: 0.1839395
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 293.92
Output dim: 4, lower bound: -0.1842853, upper bound: 0.1842562
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 293.92
Output dim: 4, lower bound: -0.1844311, upper bound: 0.1839340
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 293.92
Output dim: 4, lower bound: -0.1844452, upper bound: 0.1842455
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 293.92
Output dim: 4, lower bound: -0.1842760, upper bound: 0.1841353
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 293.92
Output dim: 4, lower bound: -0.1842873, upper bound: 0.1844535
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 293.92
Output dim: 4, lower bound: -0.1844334, upper bound: 0.1841278
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 293.92
Output dim: 4, lower bound: -0.1844448, upper bound: 0.1844447
Binary search (step 0): status=Status.UNKNOWN, k_low=2, k_high=8, k_mid=5, eps_mid=0.0195312, abs_max=0.8765864372253418
rel_dist={4: [-0.18467027294666494, 0.18467521355340155]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 290
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 290

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427664, upper bound: 0.1422525
time: 76.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427671, upper bound: 0.1427748
time: 16.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 92.82 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 92.82
Output dim: 4, lower bound: -0.1427664, upper bound: 0.1422525
IS_A2, status: Status.UNKNOWN, split count: 1, time: 92.82
Output dim: 4, lower bound: -0.1427671, upper bound: 0.1427748

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9373658, 1.8808306, 0.9349074, 1.8809935, -0.6774002, 0.6796009
1: -3.3615472, -0.0222447, -3.3618009, -0.0233908, -2.8988924, 2.9004459
2: 0.3615273, 1.0395275, 0.3618875, 1.0396043, -0.2324458, 0.2331362
3: -2.1354001, -0.3302180, -2.1359818, -0.3281253, -1.1725844, 1.1710675
4: -2.0165973, -0.6043231, -2.0228081, -0.6026021, -0.8392498, 0.8437765
5: -2.2459154, -0.3914326, -2.2467568, -0.3890823, -1.2589209, 1.2574031
6: -6.3506284, -3.0248573, -6.3527985, -3.0181296, -0.9674689, 0.9629019
7: -2.6405604, 0.3148721, -2.6430850, 0.3194143, -1.6703340, 1.6671855
8: -2.6678741, 0.1454756, -2.6741681, 0.1478813, -2.5747857, 2.5801160
9: -3.6063704, -1.2813771, -3.6069384, -1.2812705, -1.8290882, 1.8295619

Time for backsubstitution: 4.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 3068

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427096, upper bound: 0.1421350
time: 194.52 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427185, upper bound: 0.1422069
time: 94.24 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9346080, 1.8825622, 0.9346042, 1.8825649, -0.6813352, 0.6794778
1: -3.3612962, -0.0221319, -3.3616056, -0.0221317, -2.9002893, 2.9005322
2: 0.3617358, 1.0399849, 0.3617351, 1.0400261, -0.2348143, 0.2329036
3: -2.1392198, -0.3277416, -2.1392214, -0.3277414, -1.1754302, 1.1767468
4: -2.0234258, -0.5932908, -2.0234263, -0.5932882, -0.8554041, 0.8476063
5: -2.2512388, -0.3884903, -2.2512426, -0.3884900, -1.2622358, 1.2646699
6: -6.3655934, -3.0180769, -6.3656082, -3.0180769, -0.9617487, 0.9824220
7: -2.6508150, 0.3194544, -2.6508207, 0.3194550, -1.6691117, 1.6794585
8: -2.6749899, 0.1594675, -2.6749933, 0.1594725, -2.5957847, 2.5941851
9: -3.6082540, -1.2810316, -3.6083841, -1.2810123, -1.8316200, 1.8304411

Time for backsubstitution: 4.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 3068

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427158, upper bound: 0.1426455
time: 234.95 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1427161, upper bound: 0.1427260
time: 27.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 266.62 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 266.62
Output dim: 4, lower bound: -0.1427096, upper bound: 0.1421350
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 266.62
Output dim: 4, lower bound: -0.1427185, upper bound: 0.1422069
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 266.62
Output dim: 4, lower bound: -0.1427158, upper bound: 0.1426455
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 266.62
Output dim: 4, lower bound: -0.1427161, upper bound: 0.1427260

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.9374065, 1.8805076, 0.9349676, 1.8806161, -0.6769980, 0.6792171
1: -3.3610685, -0.0291276, -3.3592443, -0.0311811, -2.8908710, 2.8914518
2: 0.3623580, 1.0395266, 0.3628296, 1.0394169, -0.2315106, 0.2322508
3: -2.1291327, -0.3302197, -2.1289787, -0.3293256, -1.1648406, 1.1634870
4: -2.0135245, -0.6043350, -2.0192990, -0.6024961, -0.8357227, 0.8401811
5: -2.2390895, -0.3914400, -2.2390578, -0.3909183, -1.2504849, 1.2493451
6: -6.3447504, -3.0248590, -6.3460560, -3.0182400, -0.9598992, 0.9556071
7: -2.6350780, 0.3148615, -2.6367273, 0.3172637, -1.6629980, 1.6608444
8: -2.6677165, 0.1425022, -2.6732936, 0.1444736, -2.5713139, 2.5763891
9: -3.6060934, -1.2830145, -3.6059685, -1.2831546, -1.8266659, 1.8261939

Time for backsubstitution: 4.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425717, upper bound: 0.1420461
time: 18.90 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426685, upper bound: 0.1420464
time: 26.66 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.9373789, 1.8806663, 0.9349217, 1.8808078, -0.6771909, 0.6794627
1: -3.3614423, -0.0233014, -3.3616896, -0.0244689, -2.8963904, 2.8993137
2: 0.3616228, 1.0395273, 0.3619899, 1.0396041, -0.2323772, 0.2323321
3: -2.1340752, -0.3302191, -2.1344955, -0.3281265, -1.1719768, 1.1643778
4: -2.0152318, -0.6043249, -2.0212471, -0.6026044, -0.8383049, 0.8405950
5: -2.2446053, -0.3914359, -2.2452869, -0.3890859, -1.2583661, 1.2503035
6: -6.3493681, -3.0248582, -6.3514032, -3.0181313, -0.9665248, 0.9566223
7: -2.6394203, 0.3148693, -2.6418159, 0.3194111, -1.6700549, 1.6599092
8: -2.6678214, 0.1445230, -2.6741109, 0.1468331, -2.5735381, 2.5790448
9: -3.6062880, -1.2820880, -3.6068482, -1.2820916, -1.8283896, 1.8289716

Time for backsubstitution: 4.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425728, upper bound: 0.1421576
time: 60.40 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426688, upper bound: 0.1421676
time: 12.54 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.9346484, 1.8822395, 0.9346646, 1.8821878, -0.6809335, 0.6790943
1: -3.3608038, -0.0290174, -3.3590384, -0.0299225, -2.8922615, 2.8915315
2: 0.3625659, 1.0399842, 0.3626772, 1.0398387, -0.2338792, 0.2320180
3: -2.1329517, -0.3277429, -2.1322186, -0.3289408, -1.1676843, 1.1691664
4: -2.0203466, -0.5933032, -2.0199158, -0.5931825, -0.8518703, 0.8440052
5: -2.2444119, -0.3884979, -2.2435434, -0.3903269, -1.2537984, 1.2566116
6: -6.3597169, -3.0180793, -6.3588657, -3.0181863, -0.9541788, 0.9751271
7: -2.6453321, 0.3194440, -2.6444631, 0.3173044, -1.6617744, 1.6731124
8: -2.6748331, 0.1564942, -2.6741180, 0.1560652, -2.5923119, 2.5904572
9: -3.6079745, -1.2826694, -3.6074126, -1.2828962, -1.8291955, 1.8270712

Time for backsubstitution: 4.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425701, upper bound: 0.1425547
time: 30.64 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426697, upper bound: 0.1425568
time: 26.44 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.9346210, 1.8823980, 0.9346185, 1.8823792, -0.6811262, 0.6793397
1: -3.3611891, -0.0231876, -3.3614914, -0.0232077, -2.8977883, 2.8994002
2: 0.3618312, 1.0399848, 0.3618377, 1.0400259, -0.2347461, 0.2320992
3: -2.1378949, -0.3277428, -2.1377351, -0.3277425, -1.1748226, 1.1700553
4: -2.0220580, -0.5932934, -2.0218644, -0.5932909, -0.8544562, 0.8444209
5: -2.2499282, -0.3884937, -2.2497725, -0.3884941, -1.2616811, 1.2575686
6: -6.3643346, -3.0180779, -6.3642135, -3.0180779, -0.9608045, 0.9761420
7: -2.6496754, 0.3194516, -2.6495516, 0.3194519, -1.6688323, 1.6721783
8: -2.6749372, 0.1585146, -2.6749353, 0.1584245, -2.5945370, 2.5931122
9: -3.6081710, -1.2817427, -3.6082938, -1.2818336, -1.8309212, 1.8298507

Time for backsubstitution: 4.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425730, upper bound: 0.1426781
time: 19.85 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426669, upper bound: 0.1426775
time: 15.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 39.44 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 39.44
Output dim: 4, lower bound: -0.1425717, upper bound: 0.1420461
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 39.44
Output dim: 4, lower bound: -0.1426685, upper bound: 0.1420464
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 39.44
Output dim: 4, lower bound: -0.1425728, upper bound: 0.1421576
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 39.44
Output dim: 4, lower bound: -0.1426688, upper bound: 0.1421676
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 39.44
Output dim: 4, lower bound: -0.1425701, upper bound: 0.1425547
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 39.44
Output dim: 4, lower bound: -0.1426697, upper bound: 0.1425568
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 39.44
Output dim: 4, lower bound: -0.1425730, upper bound: 0.1426781
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 39.44
Output dim: 4, lower bound: -0.1426669, upper bound: 0.1426775

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9374436, 1.8801484, 0.9349998, 1.8803089, -0.6766566, 0.6788291
1: -3.3606637, -0.0347230, -3.3588960, -0.0359814, -2.8857684, 2.8855910
2: 0.3629550, 1.0395262, 0.3633410, 1.0394164, -0.2308251, 0.2316542
3: -2.1230738, -0.3302208, -2.1237845, -0.3293266, -1.1589472, 1.1584347
4: -2.0108538, -0.6043456, -2.0170035, -0.6025048, -0.8331015, 0.8379257
5: -2.2331641, -0.3914463, -2.2339783, -0.3909238, -1.2443295, 1.2440612
6: -6.3400278, -3.0248606, -6.3418918, -3.0182405, -0.9544277, 0.9508280
7: -2.6302619, 0.3148526, -2.6325994, 0.3172559, -1.6582344, 1.6567533
8: -2.6675701, 0.1397179, -2.6731687, 0.1420861, -2.5688324, 2.5735199
9: -3.6058426, -1.2847105, -3.6057529, -1.2846087, -1.8251545, 1.8244505

Time for backsubstitution: 4.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425193, upper bound: 0.1418072
time: 24.53 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425327, upper bound: 0.1419929
time: 244.38 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9372824, 1.8801410, 0.9350020, 1.8802972, -0.6768402, 0.6788098
1: -3.3686712, -0.0339108, -3.3589158, -0.0357487, -2.8947756, 2.8862963
2: 0.3628258, 1.0399133, 0.3633204, 1.0394166, -0.2308917, 0.2326569
3: -2.1233051, -0.3265369, -2.1237803, -0.3293265, -1.1591892, 1.1674459
4: -2.0111675, -0.6029203, -2.0171928, -0.6025045, -0.8334662, 0.8413712
5: -2.2333550, -0.3879310, -2.2338657, -0.3909237, -1.2446177, 1.2528363
6: -6.3412623, -3.0217614, -6.3428774, -3.0182407, -0.9546897, 0.9583029
7: -2.6302147, 0.3187492, -2.6324701, 0.3172567, -1.6583931, 1.6656221
8: -2.6690450, 0.1395957, -2.6731608, 0.1419362, -2.5701430, 2.5734339
9: -3.6083314, -1.2847090, -3.6057401, -1.2846336, -1.8273063, 1.8243585

Time for backsubstitution: 4.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426158, upper bound: 0.1418111
time: 22.90 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426293, upper bound: 0.1419960
time: 198.27 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9374177, 1.8803062, 0.9349555, 1.8804986, -0.6768380, 0.6790536
1: -3.3609853, -0.0300331, -3.3612826, -0.0304108, -2.8901160, 2.8921554
2: 0.3623327, 1.0395266, 0.3626112, 1.0396036, -0.2314878, 0.2316713
3: -2.1272740, -0.3302201, -2.1285148, -0.3281275, -1.1643376, 1.1591440
4: -2.0121508, -0.6043373, -2.0185812, -0.6026150, -0.8349628, 0.8381640
5: -2.2379711, -0.3914425, -2.2394333, -0.3890915, -1.2504405, 1.2447819
6: -6.3438497, -3.0248606, -6.3465219, -3.0181327, -0.9594806, 0.9514225
7: -2.6341016, 0.3148592, -2.6370678, 0.3194020, -1.6636603, 1.6558179
8: -2.6676736, 0.1416511, -2.6739800, 0.1442930, -2.5709839, 2.5761518
9: -3.6060271, -1.2840044, -3.6066153, -1.2837449, -1.8267806, 1.8271606

Time for backsubstitution: 4.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425190, upper bound: 0.1419275
time: 194.43 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425298, upper bound: 0.1421243
time: 16.74 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9372435, 1.8803868, 0.9349462, 1.8805681, -0.6771126, 0.6791537
1: -3.3692048, -0.0255895, -3.3615048, -0.0267231, -2.9025991, 2.8965838
2: 0.3618245, 1.0399144, 0.3622304, 1.0396038, -0.2317937, 0.2326799
3: -2.1307447, -0.3265364, -2.1315281, -0.3281272, -1.1662382, 1.1675947
4: -2.0139508, -0.6029061, -2.0200963, -0.6026090, -0.8361597, 0.8415779
5: -2.2413177, -0.3879260, -2.2422752, -0.3890898, -1.2525578, 1.2532043
6: -6.3481951, -3.0217612, -6.3502989, -3.0181327, -0.9616200, 0.9592904
7: -2.6364610, 0.3187604, -2.6392415, 0.3194072, -1.6646514, 1.6630321
8: -2.6691911, 0.1424331, -2.6740124, 0.1449967, -2.5730193, 2.5768702
9: -3.6086068, -1.2831926, -3.6066890, -1.2830517, -1.8294677, 1.8276495

Time for backsubstitution: 4.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426170, upper bound: 0.1419278
time: 233.68 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426289, upper bound: 0.1421214
time: 21.46 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9346858, 1.8818804, 0.9346964, 1.8818798, -0.6805929, 0.6787066
1: -3.3603880, -0.0346208, -3.3586814, -0.0347269, -2.8871522, 2.8856680
2: 0.3631626, 1.0399837, 0.3631887, 1.0398383, -0.2331938, 0.2314213
3: -2.1268928, -0.3277443, -2.1270242, -0.3289417, -1.1617912, 1.1641146
4: -2.0176711, -0.5933143, -2.0176182, -0.5931914, -0.8492401, 0.8417469
5: -2.2384858, -0.3885041, -2.2384636, -0.3903323, -1.2476428, 1.2513278
6: -6.3549943, -3.0180805, -6.3547025, -3.0181873, -0.9487076, 0.9703487
7: -2.6405160, 0.3194350, -2.6403356, 0.3172973, -1.6570086, 1.6690212
8: -2.6746869, 0.1537092, -2.6739931, 0.1536775, -2.5898311, 2.5875871
9: -3.6077204, -1.2843658, -3.6071954, -1.2843504, -1.8276825, 1.8253266

Time for backsubstitution: 4.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425198, upper bound: 0.1423197
time: 17.00 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425300, upper bound: 0.1425201
time: 11.14 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9345238, 1.8818734, 0.9346983, 1.8818686, -0.6807767, 0.6786873
1: -3.3683932, -0.0338018, -3.3587022, -0.0344896, -2.8961592, 2.8863740
2: 0.3630333, 1.0403711, 0.3631680, 1.0398383, -0.2332605, 0.2324240
3: -2.1271243, -0.3240604, -2.1270199, -0.3289419, -1.1620337, 1.1731253
4: -2.0179861, -0.5918888, -2.0178077, -0.5931905, -0.8496068, 0.8451923
5: -2.2386775, -0.3849888, -2.2383509, -0.3903324, -1.2479304, 1.2601030
6: -6.3562293, -3.0149803, -6.3556871, -3.0181875, -0.9489694, 0.9778231
7: -2.6404693, 0.3233309, -2.6402059, 0.3172975, -1.6571693, 1.6778895
8: -2.6761627, 0.1535865, -2.6739860, 0.1535265, -2.5911422, 2.5875020
9: -3.6102097, -1.2843635, -3.6071825, -1.2843754, -1.8298341, 1.8252344

Time for backsubstitution: 4.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426162, upper bound: 0.1423218
time: 23.34 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426266, upper bound: 0.1425097
time: 143.69 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9346595, 1.8820384, 0.9346524, 1.8820701, -0.6807737, 0.6789309
1: -3.3607187, -0.0299263, -3.3610754, -0.0291538, -2.8915079, 2.8922386
2: 0.3625406, 1.0399841, 0.3624590, 1.0400254, -0.2338570, 0.2314386
3: -2.1310937, -0.3277436, -2.1317549, -0.3277434, -1.1671839, 1.1648217
4: -2.0189695, -0.5933052, -2.0191960, -0.5933014, -0.8511055, 0.8419868
5: -2.2432942, -0.3885000, -2.2439191, -0.3884995, -1.2537546, 1.2520466
6: -6.3588157, -3.0180798, -6.3593321, -3.0180795, -0.9537604, 0.9709432
7: -2.6443563, 0.3194414, -2.6448026, 0.3194429, -1.6624370, 1.6680858
8: -2.6747894, 0.1556437, -2.6748052, 0.1558849, -2.5919831, 2.5902202
9: -3.6079066, -1.2836597, -3.6080587, -1.2834866, -1.8293107, 1.8280380

Time for backsubstitution: 4.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425209, upper bound: 0.1424432
time: 244.84 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425294, upper bound: 0.1426288
time: 187.99 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9344853, 1.8821186, 0.9346426, 1.8821398, -0.6810484, 0.6790308
1: -3.3689427, -0.0254753, -3.3613050, -0.0254619, -2.9039936, 2.8966711
2: 0.3620326, 1.0403720, 0.3620784, 1.0400257, -0.2341628, 0.2324469
3: -2.1345642, -0.3240597, -2.1347680, -0.3277431, -1.1690849, 1.1732725
4: -2.0207751, -0.5918742, -2.0207131, -0.5932951, -0.8523101, 0.8454025
5: -2.2466409, -0.3849834, -2.2467606, -0.3884985, -1.2558722, 1.2604694
6: -6.3631616, -3.0149801, -6.3631091, -3.0180793, -0.9558997, 0.9788106
7: -2.6467147, 0.3233421, -2.6469774, 0.3194471, -1.6634291, 1.6753000
8: -2.6763077, 0.1564247, -2.6748383, 0.1565884, -2.5940192, 2.5909393
9: -3.6104872, -1.2828479, -3.6081328, -1.2827934, -1.8319972, 1.8285272

Time for backsubstitution: 4.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426152, upper bound: 0.1424387
time: 77.56 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426303, upper bound: 0.1426405
time: 16.81 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 98.82 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 98.82
Output dim: 4, lower bound: -0.1425193, upper bound: 0.1418072
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 98.82
Output dim: 4, lower bound: -0.1425327, upper bound: 0.1419929
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 98.82
Output dim: 4, lower bound: -0.1426158, upper bound: 0.1418111
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 98.82
Output dim: 4, lower bound: -0.1426293, upper bound: 0.1419960
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 98.82
Output dim: 4, lower bound: -0.1425190, upper bound: 0.1419275
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 98.82
Output dim: 4, lower bound: -0.1425298, upper bound: 0.1421243
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 98.82
Output dim: 4, lower bound: -0.1426170, upper bound: 0.1419278
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 98.82
Output dim: 4, lower bound: -0.1426289, upper bound: 0.1421214
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 98.82
Output dim: 4, lower bound: -0.1425198, upper bound: 0.1423197
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 98.82
Output dim: 4, lower bound: -0.1425300, upper bound: 0.1425201
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 98.82
Output dim: 4, lower bound: -0.1426162, upper bound: 0.1423218
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 98.82
Output dim: 4, lower bound: -0.1426266, upper bound: 0.1425097
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 98.82
Output dim: 4, lower bound: -0.1425209, upper bound: 0.1424432
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 98.82
Output dim: 4, lower bound: -0.1425294, upper bound: 0.1426288
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 98.82
Output dim: 4, lower bound: -0.1426152, upper bound: 0.1424387
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 98.82
Output dim: 4, lower bound: -0.1426303, upper bound: 0.1426405

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9375756, 1.8789920, 0.9355957, 1.8789475, -0.6751414, 0.6770632
1: -3.3595455, -0.0433457, -3.3515191, -0.0461252, -2.8747997, 2.8697584
2: 0.3633387, 1.0391004, 0.3638214, 1.0389084, -0.2298896, 0.2306533
3: -2.1100149, -0.3302320, -2.1084120, -0.3332646, -1.1391194, 1.1416651
4: -2.0073552, -0.6043658, -2.0128644, -0.6034876, -0.8281600, 0.8335176
5: -2.2196193, -0.3914917, -2.2179437, -0.3952059, -1.2235267, 1.2260029
6: -6.3330941, -3.0248623, -6.3336911, -3.0184426, -0.9473746, 0.9429139
7: -2.6173100, 0.3148319, -2.6174583, 0.3132965, -1.6347148, 1.6383650
8: -2.6670370, 0.1302333, -2.6694520, 0.1309825, -2.5578547, 2.5607271
9: -3.6047788, -1.2906166, -3.5980520, -1.2915506, -1.8185866, 1.8123260

Time for backsubstitution: 4.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 2390

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1423077, upper bound: 0.1417755
time: 12.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425142, upper bound: 0.1417854
time: 13.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9374614, 1.8799909, 0.9350193, 1.8801301, -0.6763817, 0.6786531
1: -3.3605227, -0.0360069, -3.3587370, -0.0374188, -2.8839388, 2.8842566
2: 0.3630039, 1.0395043, 0.3633949, 1.0393907, -0.2307311, 0.2315685
3: -2.1212533, -0.3302220, -2.1217387, -0.3293278, -1.1583109, 1.1456721
4: -2.0103304, -0.6043485, -2.0164227, -0.6025087, -0.8328481, 0.8344581
5: -2.2310724, -0.3914514, -2.2316194, -0.3909291, -1.2434473, 1.2312152
6: -6.3393707, -3.0248606, -6.3411551, -3.0182405, -0.9536538, 0.9491038
7: -2.6282134, 0.3148493, -2.6303155, 0.3172535, -1.6580760, 1.6403444
8: -2.6674967, 0.1383882, -2.6730883, 0.1406257, -2.5671101, 2.5719779
9: -3.6057127, -1.2853385, -3.6056080, -1.2852788, -1.8241321, 1.8235469

Time for backsubstitution: 4.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2390

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1423124, upper bound: 0.1419838
time: 31.16 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425278, upper bound: 0.1420002
time: 38.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9374141, 1.8789845, 0.9355974, 1.8789363, -0.6753249, 0.6770437
1: -3.3675525, -0.0425334, -3.3515389, -0.0458918, -2.8838081, 2.8704648
2: 0.3632101, 1.0394878, 0.3638010, 1.0389084, -0.2299561, 0.2316562
3: -2.1102464, -0.3265480, -2.1084082, -0.3332644, -1.1393622, 1.1506774
4: -2.0076685, -0.6029399, -2.0130537, -0.6034864, -0.8285239, 0.8369644
5: -2.2198091, -0.3879769, -2.2178311, -0.3952059, -1.2238196, 1.2347794
6: -6.3343277, -3.0217628, -6.3346734, -3.0184422, -0.9476364, 0.9503905
7: -2.6172636, 0.3187286, -2.6173289, 0.3132973, -1.6348741, 1.6472340
8: -2.6685119, 0.1301111, -2.6694443, 0.1308317, -2.5591660, 2.5606411
9: -3.6072683, -1.2906148, -3.5980392, -1.2915754, -1.8207384, 1.8122343

Time for backsubstitution: 4.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2390

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1424018, upper bound: 0.1417721
time: 19.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426087, upper bound: 0.1417797
time: 16.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9372999, 1.8799835, 0.9350211, 1.8801188, -0.6765654, 0.6786336
1: -3.3685298, -0.0351942, -3.3587570, -0.0371854, -2.8929465, 2.8849621
2: 0.3628746, 1.0398916, 0.3633741, 1.0393908, -0.2307976, 0.2325713
3: -2.1214848, -0.3265378, -2.1217346, -0.3293277, -1.1585529, 1.1546829
4: -2.0106432, -0.6029234, -2.0166116, -0.6025077, -0.8332130, 0.8379041
5: -2.2312639, -0.3879362, -2.2315066, -0.3909292, -1.2437358, 1.2399907
6: -6.3406053, -3.0217614, -6.3421392, -3.0182405, -0.9539162, 0.9565784
7: -2.6281655, 0.3187464, -2.6301858, 0.3172537, -1.6582351, 1.6492130
8: -2.6689723, 0.1382660, -2.6730802, 0.1404759, -2.5684214, 2.5718918
9: -3.6082020, -1.2853367, -3.6055944, -1.2853042, -1.8262845, 1.8234547

Time for backsubstitution: 4.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2390

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1424106, upper bound: 0.1419823
time: 191.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1426249, upper bound: 0.1419909
time: 184.96 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9375492, 1.8791496, 0.9355519, 1.8791379, -0.6753224, 0.6772870
1: -3.3598697, -0.0386572, -3.3539102, -0.0405562, -2.8791492, 2.8763258
2: 0.3627167, 1.0391009, 0.3630923, 1.0390954, -0.2305529, 0.2306715
3: -2.1142156, -0.3302314, -2.1131408, -0.3320656, -1.1445092, 1.1423767
4: -2.0086503, -0.6043568, -2.0144401, -0.6035974, -0.8300223, 0.8337554
5: -2.2244220, -0.3914878, -2.2233927, -0.3933737, -1.2296276, 1.2267165
6: -6.3369112, -3.0248618, -6.3383160, -3.0183346, -0.9524295, 0.9435142
7: -2.6211503, 0.3148384, -2.6219263, 0.3154427, -1.6401416, 1.6374297
8: -2.6671417, 0.1321665, -2.6702638, 0.1331876, -2.5600042, 2.5633585
9: -3.6049647, -1.2899102, -3.5989161, -1.2906871, -1.8202133, 1.8150375

Time for backsubstitution: 4.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2390

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1423064, upper bound: 0.1418894
time: 147.99 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425149, upper bound: 0.1418916
time: 152.86 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9374354, 1.8801484, 0.9349751, 1.8803203, -0.6765629, 0.6788773
1: -3.3608451, -0.0313172, -3.3611250, -0.0318484, -2.8882875, 2.8908219
2: 0.3623819, 1.0395048, 0.3626652, 1.0395778, -0.2313939, 0.2315858
3: -2.1254544, -0.3302213, -2.1264691, -0.3281285, -1.1637017, 1.1463819
4: -2.0116262, -0.6043398, -2.0180001, -0.6026185, -0.8347101, 0.8346976
5: -2.2358797, -0.3914469, -2.2370739, -0.3890965, -1.2495587, 1.2319365
6: -6.3431907, -3.0248601, -6.3457847, -3.0181327, -0.9587066, 0.9496992
7: -2.6320524, 0.3148569, -2.6347828, 0.3193992, -1.6635027, 1.6394088
8: -2.6676013, 0.1403217, -2.6738997, 0.1428332, -2.5692620, 2.5746105
9: -3.6058965, -1.2846322, -3.6064706, -1.2844152, -1.8257593, 1.8262572

Time for backsubstitution: 4.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 2390

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1423164, upper bound: 0.1420999
time: 390.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1425274, upper bound: 0.1421128
time: 166.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 561.92 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 561.92
Output dim: 4, lower bound: -0.1423077, upper bound: 0.1417755
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 561.92
Output dim: 4, lower bound: -0.1425142, upper bound: 0.1417854
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 561.92
Output dim: 4, lower bound: -0.1423124, upper bound: 0.1419838
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 561.92
Output dim: 4, lower bound: -0.1425278, upper bound: 0.1420002
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 561.92
Output dim: 4, lower bound: -0.1424018, upper bound: 0.1417721
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 561.92
Output dim: 4, lower bound: -0.1426087, upper bound: 0.1417797
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 561.92
Output dim: 4, lower bound: -0.1424106, upper bound: 0.1419823
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 561.92
Output dim: 4, lower bound: -0.1426249, upper bound: 0.1419909
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 561.92
Output dim: 4, lower bound: -0.1423064, upper bound: 0.1418894
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 561.92
Output dim: 4, lower bound: -0.1425149, upper bound: 0.1418916
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 561.92
Output dim: 4, lower bound: -0.1423164, upper bound: 0.1420999
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 561.92
Output dim: 4, lower bound: -0.1425274, upper bound: 0.1421128
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 561.92
Output dim: 4, lower bound: -0.1426170, upper bound: 0.1419278
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 561.92
Output dim: 4, lower bound: -0.1426289, upper bound: 0.1421214
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 561.92
Output dim: 4, lower bound: -0.1425198, upper bound: 0.1423197
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 561.92
Output dim: 4, lower bound: -0.1425300, upper bound: 0.1425201
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 561.92
Output dim: 4, lower bound: -0.1426162, upper bound: 0.1423218
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 561.92
Output dim: 4, lower bound: -0.1426266, upper bound: 0.1425097
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 561.92
Output dim: 4, lower bound: -0.1425209, upper bound: 0.1424432
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 561.92
Output dim: 4, lower bound: -0.1425294, upper bound: 0.1426288
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 561.92
Output dim: 4, lower bound: -0.1426152, upper bound: 0.1424387
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 561.92
Output dim: 4, lower bound: -0.1426303, upper bound: 0.1426405
Binary search (step 1): status=Status.UNKNOWN, k_low=2, k_high=4, k_mid=3, eps_mid=0.0117188, abs_max=0.8554123640060425
rel_dist={4: [-0.1427653512292979, 0.1427739341779526]}

## Binary search (step 2) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 290
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 290

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216686, upper bound: 0.1213470
time: 40.91 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216678, upper bound: 0.1216864
time: 22.97 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 64.01 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 64.01
Output dim: 4, lower bound: -0.1216686, upper bound: 0.1213470
IS_A2, status: Status.UNKNOWN, split count: 1, time: 64.01
Output dim: 4, lower bound: -0.1216678, upper bound: 0.1216864

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9373658, 1.8808306, 0.9349249, 1.8809463, -0.6716160, 0.6738551
1: -3.3615472, -0.0222447, -3.3617551, -0.0234571, -2.8871334, 2.8886857
2: 0.3615273, 1.0395275, 0.3618963, 1.0395817, -0.2259152, 0.2266775
3: -2.1354001, -0.3302180, -2.1357954, -0.3281453, -1.1618702, 1.1601796
4: -2.0165973, -0.6043231, -2.0227735, -0.6031386, -0.8281248, 0.8331600
5: -2.2459154, -0.3914326, -2.2465000, -0.3891163, -1.2472790, 1.2455174
6: -6.3506284, -3.0248573, -6.3520575, -3.0181327, -0.9335073, 0.9282064
7: -2.6405604, 0.3148721, -2.6426435, 0.3194124, -1.6489872, 1.6455460
8: -2.6678741, 0.1454756, -2.6741209, 0.1472136, -2.5671396, 2.5731621
9: -3.6063704, -1.2813771, -3.6068521, -1.2812908, -1.8158380, 1.8162996

Time for backsubstitution: 4.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 3068

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216350, upper bound: 0.1212616
time: 18.70 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216430, upper bound: 0.1213127
time: 19.08 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9346080, 1.8825622, 0.9346055, 1.8825641, -0.6755971, 0.6737180
1: -3.3612962, -0.0221319, -3.3615029, -0.0221317, -2.8885911, 2.8887258
2: 0.3617358, 1.0399849, 0.3617354, 1.0400127, -0.2283607, 0.2264537
3: -2.1392198, -0.3277416, -2.1392212, -0.3277415, -1.1647146, 1.1660478
4: -2.0234258, -0.5932908, -2.0234263, -0.5932891, -0.8448167, 0.8369398
5: -2.2512388, -0.3884903, -2.2512417, -0.3884900, -1.2505819, 1.2530468
6: -6.3655934, -3.0180769, -6.3656030, -3.0180767, -0.9275196, 0.9484582
7: -2.6508150, 0.3194544, -2.6508183, 0.3194549, -1.6476011, 1.6581103
8: -2.6749899, 0.1594675, -2.6749916, 0.1594707, -2.5888495, 2.5872326
9: -3.6082540, -1.2810316, -3.6083376, -1.2810184, -1.8183542, 1.8171263

Time for backsubstitution: 4.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 3068

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216407, upper bound: 0.1215962
time: 480.62 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216361, upper bound: 0.1216289
time: 153.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 638.76 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 638.76
Output dim: 4, lower bound: -0.1216350, upper bound: 0.1212616
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 638.76
Output dim: 4, lower bound: -0.1216430, upper bound: 0.1213127
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 638.76
Output dim: 4, lower bound: -0.1216407, upper bound: 0.1215962
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 638.76
Output dim: 4, lower bound: -0.1216361, upper bound: 0.1216289

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.9374081, 1.8804915, 0.9349852, 1.8805692, -0.6712126, 0.6734560
1: -3.3610477, -0.0294120, -3.3591983, -0.0312479, -2.8790967, 2.8794065
2: 0.3623915, 1.0395267, 0.3628385, 1.0393944, -0.2249530, 0.2257921
3: -2.1289070, -0.3302198, -2.1287916, -0.3293454, -1.1538789, 1.1525996
4: -2.0133884, -0.6043354, -2.0192649, -0.6030324, -0.8244762, 0.8295639
5: -2.2388167, -0.3914404, -2.2388012, -0.3909528, -1.2385879, 1.2374588
6: -6.3444810, -3.0248594, -6.3453140, -3.0182433, -0.9257176, 0.9209111
7: -2.6348171, 0.3148610, -2.6362855, 0.3172619, -1.6414526, 1.6392047
8: -2.6677110, 0.1423698, -2.6732469, 0.1438048, -2.5636640, 2.5693030
9: -3.6060820, -1.2830876, -3.6058829, -1.2831748, -1.8134123, 1.8128381

Time for backsubstitution: 4.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1215373, upper bound: 0.1211966
time: 44.75 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216016, upper bound: 0.1211898
time: 402.42 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.9373795, 1.8806595, 0.9349391, 1.8807603, -0.6714060, 0.6737127
1: -3.3614399, -0.0233088, -3.3616431, -0.0245357, -2.8846173, 2.8875475
2: 0.3616253, 1.0395273, 0.3619988, 1.0395815, -0.2258442, 0.2258631
3: -2.1340234, -0.3302194, -2.1343088, -0.3281461, -1.1612337, 1.1534206
4: -2.0151641, -0.6043252, -2.0212123, -0.6031407, -0.8271327, 0.8299627
5: -2.2445540, -0.3914360, -2.2450304, -0.3891205, -1.2466981, 1.2383363
6: -6.3493142, -3.0248590, -6.3506622, -3.0181339, -0.9325355, 0.9218500
7: -2.6393793, 0.3148693, -2.6413748, 0.3194090, -1.6486944, 1.6381798
8: -2.6678195, 0.1444927, -2.6740630, 0.1461647, -2.5658901, 2.5720580
9: -3.6062853, -1.2821225, -3.6067617, -1.2821120, -1.8151387, 1.8156912

Time for backsubstitution: 4.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1215365, upper bound: 0.1212878
time: 14.01 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216025, upper bound: 0.1212699
time: 475.92 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.9346501, 1.8822230, 0.9346658, 1.8821868, -0.6751941, 0.6733193
1: -3.3607829, -0.0293014, -3.3589354, -0.0299230, -2.8805459, 2.8794410
2: 0.3625994, 1.0399842, 0.3626777, 1.0398256, -0.2273985, 0.2255680
3: -2.1327262, -0.3277430, -2.1322176, -0.3289408, -1.1567214, 1.1584673
4: -2.0202100, -0.5933042, -2.0199153, -0.5931835, -0.8411610, 0.8333379
5: -2.2441397, -0.3884985, -2.2435417, -0.3903271, -1.2418890, 1.2449884
6: -6.3594470, -3.0180793, -6.3588624, -3.0181870, -0.9197297, 0.9411637
7: -2.6450717, 0.3194436, -2.6444609, 0.3173044, -1.6400645, 1.6517639
8: -2.6748269, 0.1563618, -2.6741176, 0.1560630, -2.5853739, 2.5833716
9: -3.6079628, -1.2827432, -3.6073666, -1.2829027, -1.8159274, 1.8136629

Time for backsubstitution: 4.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1215375, upper bound: 0.1215250
time: 199.31 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216066, upper bound: 0.1215429
time: 16.19 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.9346214, 1.8823913, 0.9346200, 1.8823781, -0.6753876, 0.6735758
1: -3.3611860, -0.0231936, -3.3613887, -0.0232081, -2.8860760, 2.8875875
2: 0.3618334, 1.0399848, 0.3618379, 1.0400127, -0.2282900, 0.2256388
3: -2.1378429, -0.3277427, -2.1377344, -0.3277425, -1.1640776, 1.1592872
4: -2.0219893, -0.5932929, -2.0218644, -0.5932916, -0.8438222, 0.8337383
5: -2.2498772, -0.3884940, -2.2497718, -0.3884943, -1.2500011, 1.2458642
6: -6.3642793, -3.0180774, -6.3642092, -3.0180779, -0.9265478, 0.9421022
7: -2.6496339, 0.3194517, -2.6495490, 0.3194520, -1.6473079, 1.6507403
8: -2.6749353, 0.1584838, -2.6749346, 0.1584223, -2.5875998, 2.5861270
9: -3.6081686, -1.2817779, -3.6082468, -1.2818398, -1.8176544, 1.8165165

Time for backsubstitution: 4.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 3069

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215453, upper bound: 0.1216172
time: 156.25 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1216067, upper bound: 0.1216235
time: 11.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 172.31 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 172.31
Output dim: 4, lower bound: -0.1215373, upper bound: 0.1211966
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 172.31
Output dim: 4, lower bound: -0.1216016, upper bound: 0.1211898
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 172.31
Output dim: 4, lower bound: -0.1215365, upper bound: 0.1212878
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 172.31
Output dim: 4, lower bound: -0.1216025, upper bound: 0.1212699
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 172.31
Output dim: 4, lower bound: -0.1215375, upper bound: 0.1215250
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 172.31
Output dim: 4, lower bound: -0.1216066, upper bound: 0.1215429
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 172.31
Output dim: 4, lower bound: -0.1215453, upper bound: 0.1216172
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 172.31
Output dim: 4, lower bound: -0.1216067, upper bound: 0.1216235

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9372837, 1.8801252, 0.9350209, 1.8802338, -0.6710377, 0.6730475
1: -3.3686459, -0.0343111, -3.3588545, -0.0359533, -2.8828573, 2.8741398
2: 0.3628681, 1.0399134, 0.3633455, 1.0393939, -0.2243294, 0.2261914
3: -2.1230788, -0.3265370, -2.1233447, -0.3293463, -1.1481740, 1.1564924
4: -2.0110250, -0.6029210, -2.0170608, -0.6030408, -0.8221991, 0.8307160
5: -2.2330830, -0.3879313, -2.2333574, -0.3909584, -1.2326669, 1.2408803
6: -6.3409925, -3.0217621, -6.3420081, -3.0182438, -0.9204853, 0.9235473
7: -2.6299553, 0.3187482, -2.6318095, 0.3172540, -1.6367776, 1.6439412
8: -2.6690388, 0.1394638, -2.6731062, 0.1411396, -2.5623658, 2.5663445
9: -3.6083202, -1.2847824, -3.6056426, -1.2847294, -1.8139740, 1.8109994

Time for backsubstitution: 4.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215679, upper bound: 0.1210405
time: 171.93 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215738, upper bound: 0.1211735
time: 20.24 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9372436, 1.8803847, 0.9349644, 1.8805115, -0.6713180, 0.6734052
1: -3.3692036, -0.0255947, -3.3614526, -0.0268092, -2.8908067, 2.8848109
2: 0.3618256, 1.0399144, 0.3622424, 1.0395814, -0.2252574, 0.2262081
3: -2.1307330, -0.3265362, -2.1312160, -0.3281470, -1.1554512, 1.1565957
4: -2.0139058, -0.6029061, -2.0200193, -0.6031449, -0.8249800, 0.8309321
5: -2.2413054, -0.3879260, -2.2418880, -0.3891245, -1.2408476, 1.2411942
6: -6.3481836, -3.0217605, -6.3495364, -3.0181358, -0.9276232, 0.9244981
7: -2.6364484, 0.3187599, -2.6386778, 0.3194042, -1.6432223, 1.6412630
8: -2.6691899, 0.1424188, -2.6739612, 0.1442529, -2.5652976, 2.5698979
9: -3.6086054, -1.2832174, -3.6065962, -1.2831161, -1.8161706, 1.8143764

Time for backsubstitution: 4.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1215616, upper bound: 0.1211171
time: 339.61 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215730, upper bound: 0.1212474
time: 287.30 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9345252, 1.8818570, 0.9347016, 1.8818514, -0.6750202, 0.6729110
1: -3.3683677, -0.0342019, -3.3585851, -0.0346277, -2.8843019, 2.8741717
2: 0.3630754, 1.0403709, 0.3631845, 1.0398251, -0.2267751, 0.2259671
3: -2.1268978, -0.3240606, -2.1267703, -0.3289419, -1.1510168, 1.1623602
4: -2.0178437, -0.5918897, -2.0177095, -0.5931921, -0.8388762, 0.8344874
5: -2.2384055, -0.3849891, -2.2380981, -0.3903325, -1.2359684, 1.2484095
6: -6.3559599, -3.0149808, -6.3555555, -3.0181875, -0.9144979, 0.9438000
7: -2.6402094, 0.3233306, -2.6399848, 0.3172968, -1.6353893, 1.6565001
8: -2.6761568, 0.1534551, -2.6739781, 0.1533961, -2.5840759, 2.5804133
9: -3.6101975, -1.2844374, -3.6071246, -1.2844572, -1.8164871, 1.8118219

Time for backsubstitution: 4.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215638, upper bound: 0.1213782
time: 243.99 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215801, upper bound: 0.1215169
time: 13.71 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9346603, 1.8820312, 0.9346558, 1.8820529, -0.6750184, 0.6731625
1: -3.3607137, -0.0299754, -3.3609526, -0.0294490, -2.8794978, 2.8803675
2: 0.3625463, 1.0399841, 0.3624904, 1.0400121, -0.2274008, 0.2249480
3: -2.1310003, -0.3277434, -2.1314363, -0.3277436, -1.1564095, 1.1537771
4: -2.0189011, -0.5933051, -2.0190561, -0.5933031, -0.8404703, 0.8311777
5: -2.2432013, -0.3885001, -2.2436070, -0.3884997, -1.2420449, 1.2400551
6: -6.3587608, -3.0180793, -6.3591037, -3.0180793, -0.9195036, 0.9366663
7: -2.6442745, 0.3194415, -2.6445751, 0.3194422, -1.6408834, 1.6464319
8: -2.6747870, 0.1555908, -2.6747971, 0.1557558, -2.5849180, 2.5832114
9: -3.6079023, -1.2837064, -3.6080010, -1.2835802, -1.8159666, 1.8146878

Time for backsubstitution: 4.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1215034, upper bound: 0.1214581
time: 158.33 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215091, upper bound: 0.1215879
time: 51.46 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9344855, 1.8821164, 0.9346449, 1.8821292, -0.6753002, 0.6732688
1: -3.3689427, -0.0254819, -3.3611968, -0.0254805, -2.8922625, 2.8848507
2: 0.3620336, 1.0403720, 0.3620816, 1.0400125, -0.2277034, 0.2259838
3: -2.1345522, -0.3240598, -2.1346416, -0.3277433, -1.1582955, 1.1624626
4: -2.0207303, -0.5918742, -2.0206709, -0.5932959, -0.8416672, 0.8347069
5: -2.2466280, -0.3849836, -2.2466292, -0.3884985, -1.2441494, 1.2487218
6: -6.3631492, -3.0149803, -6.3630829, -3.0180798, -0.9216354, 0.9447507
7: -2.6467028, 0.3233418, -2.6468530, 0.3194470, -1.6418359, 1.6538217
8: -2.6763072, 0.1564101, -2.6748326, 0.1565107, -2.5870068, 2.5839672
9: -3.6104860, -1.2828727, -3.6080801, -1.2828437, -1.8186846, 1.8152010

Time for backsubstitution: 4.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 2390
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 354
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2542
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2168
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 3518
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2148
type: B, layer: 1, pos: 2167
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2194
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 290
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 141
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 2816
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 2285
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2440
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 2527
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 2883
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2171
type: B, layer: 1, pos: 3131
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 2319
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2439
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2861
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2096
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 3062
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 2164
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2892
type: B, layer: 1, pos: 2500
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 2578
type: B, layer: 1, pos: 2094
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2884
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2665
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 2906
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2963
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2886
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2885
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 2890
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3132
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 2990
type: B, layer: 1, pos: 2904
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 2887
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2443
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 2903
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 206
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2288
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 2320
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2888
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 3279
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2894
type: B, layer: 1, pos: 2922
type: B, layer: 1, pos: 2923
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2389

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215627, upper bound: 0.1214647
time: 17.42 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215733, upper bound: 0.1215769
time: 65.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 87.25 seconds
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 87.25
Output dim: 4, lower bound: -0.1215679, upper bound: 0.1210405
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 87.25
Output dim: 4, lower bound: -0.1215738, upper bound: 0.1211735
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 87.25
Output dim: 4, lower bound: -0.1215616, upper bound: 0.1211171
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 87.25
Output dim: 4, lower bound: -0.1215730, upper bound: 0.1212474
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 87.25
Output dim: 4, lower bound: -0.1215638, upper bound: 0.1213782
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 87.25
Output dim: 4, lower bound: -0.1215801, upper bound: 0.1215169
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 87.25
Output dim: 4, lower bound: -0.1215034, upper bound: 0.1214581
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 87.25
Output dim: 4, lower bound: -0.1215091, upper bound: 0.1215879
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 87.25
Output dim: 4, lower bound: -0.1215627, upper bound: 0.1214647
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 87.25
Output dim: 4, lower bound: -0.1215733, upper bound: 0.1215769

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9374228, 1.8789091, 0.9356169, 1.8788730, -0.6695166, 0.6712202
1: -3.3674684, -0.0433490, -3.3514781, -0.0460975, -2.8718500, 2.8578644
2: 0.3632726, 1.0394653, 0.3638260, 1.0388858, -0.2233705, 0.2251701
3: -2.1093440, -0.3265484, -2.1079731, -0.3332840, -1.1276650, 1.1397240
4: -2.0073433, -0.6029418, -2.0129213, -0.6040232, -0.8170846, 0.8263092
5: -2.2188363, -0.3879791, -2.2173233, -0.3952408, -1.2111380, 1.2228224
6: -6.3336987, -3.0217633, -6.3338051, -3.0184450, -0.9130914, 0.9156346
7: -2.6163378, 0.3187268, -2.6166687, 0.3132941, -1.6124718, 1.6255517
8: -2.6684787, 0.1294904, -2.6693890, 0.1300358, -2.5513737, 2.5530787
9: -3.6072011, -1.2909837, -3.5979419, -1.2916715, -1.8073925, 1.7985966

Time for backsubstitution: 4.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2390

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1214205, upper bound: 0.1210004
time: 174.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1215613, upper bound: 0.1210228
time: 33.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9373020, 1.8799608, 0.9350406, 1.8800554, -0.6707615, 0.6728662
1: -3.3684995, -0.0356443, -3.3586962, -0.0373912, -2.8810234, 2.8727567
2: 0.3629189, 1.0398905, 0.3633992, 1.0393680, -0.2242334, 0.2261045
3: -2.1211863, -0.3265380, -2.1212988, -0.3293475, -1.1475092, 1.1436192
4: -2.0104795, -0.6029236, -2.0164802, -0.6030440, -0.8219340, 0.8272181
5: -2.2309053, -0.3879365, -2.2309990, -0.3909633, -1.2317439, 1.2279217
6: -6.3403091, -3.0217621, -6.3412704, -3.0182438, -0.9196896, 0.9218162
7: -2.6278300, 0.3187459, -2.6295252, 0.3172510, -1.6366113, 1.6274233
8: -2.6689637, 0.1380917, -2.6730263, 0.1396788, -2.5606408, 2.5647497
9: -3.6081858, -1.2854241, -3.6054974, -1.2853994, -1.8129506, 1.8100708

Time for backsubstitution: 4.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2390
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 354
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2542
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2168
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 3518
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2148
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2194
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 141
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 2816
type: A, layer: 1, pos: 2182
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 2285
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2440
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 2527
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 2883
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2171
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 2319
type: A, layer: 1, pos: 3131
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 2439
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 2861
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2096
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3062
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 2164
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 2892
type: A, layer: 1, pos: 2500
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 2578
type: A, layer: 1, pos: 2094
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2884
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2963
type: A, layer: 1, pos: 2906
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2665
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2890
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 2886
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 2885
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 3132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 2990
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2904
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2887
type: A, layer: 1, pos: 2443
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2903
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 206
type: A, layer: 1, pos: 2320
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2288
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2888
type: A, layer: 1, pos: 3279
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2894
type: A, layer: 1, pos: 2922
type: A, layer: 1, pos: 2923
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2390

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.1214278, upper bound: 0.1211433
time: 343.48 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.1215697, upper bound: 0.1211684
time: 143.15 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 491.01 seconds
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 491.01
Output dim: 4, lower bound: -0.1214205, upper bound: 0.1210004
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 491.01
Output dim: 4, lower bound: -0.1215613, upper bound: 0.1210228
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 491.01
Output dim: 4, lower bound: -0.1214278, upper bound: 0.1211433
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 491.01
Output dim: 4, lower bound: -0.1215697, upper bound: 0.1211684
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 491.01
Output dim: 4, lower bound: -0.1215730, upper bound: 0.1212474
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 491.01
Output dim: 4, lower bound: -0.1215638, upper bound: 0.1213782
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 491.01
Output dim: 4, lower bound: -0.1215801, upper bound: 0.1215169
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 491.01
Output dim: 4, lower bound: -0.1215091, upper bound: 0.1215879
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 491.01
Output dim: 4, lower bound: -0.1215627, upper bound: 0.1214647
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 491.01
Output dim: 4, lower bound: -0.1215733, upper bound: 0.1215769
Binary search (step 2): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=0.8448255062103271
rel_dist={4: [-0.1216752063496438, 0.12168907463758383]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 12646.06 seconds
