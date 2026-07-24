## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 7)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.0801198999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.7020321, -3.1332498, -4.7020321, -3.1332498, -1.4605174, 1.4605174)
1: (-5.4196887, -3.1153350, -5.4196887, -3.1153350, -1.5848445, 1.5848446)
2: (-0.5015193, 0.0363976, -0.5015193, 0.0363976, -0.5326717, 0.5326717)
3: (-0.7205948, -0.1012112, -0.7205948, -0.1012112, -0.5058208, 0.5058208)
4: (-0.8135126, 0.1667712, -0.8135126, 0.1667712, -0.8531543, 0.8531543)
5: (-1.1506350, -0.5262530, -1.1506350, -0.5262530, -0.3987806, 0.3987806)
6: (0.3551826, 0.5821692, 0.3551826, 0.5821692, -0.1256554, 0.1256554)
7: (-1.7589732, -0.4479041, -1.7589732, -0.4479041, -1.2075187, 1.2075186)
8: (-5.8517289, -3.9170151, -5.8517289, -3.9170151, -1.4400744, 1.4400742)
9: (-4.5360956, -2.7854774, -4.5360956, -2.7854774, -1.3318502, 1.3318503)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.80 + 342.16 = 349.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0801936, upper bound: 0.0801986

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3485
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 268
type: A, layer: 1, pos: 3026
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 3035
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 3185
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2703
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3408
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 3107
type: A, layer: 1, pos: 2705
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 2664
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 3261
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2546
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2415
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 2258
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 3293
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 3106
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 3105
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2412
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 3215
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2308
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 2865
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2447
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3155
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 2548
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3300
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2458
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 2478
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 3309
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 2820
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 3294
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2747
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 3315
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3120
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 2683
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2505
type: A, layer: 1, pos: 3424
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 2872
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 2256
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 469
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2259
type: A, layer: 1, pos: 3017
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 2900
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 2899
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2791
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 2938
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2898
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 2855
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2309
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2462
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2910
type: A, layer: 1, pos: 2912
type: A, layer: 1, pos: 2913
type: A, layer: 1, pos: 2914
type: A, layer: 1, pos: 2918
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3135
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3194
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3370

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3485

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0797637, upper bound: 0.0799331
time: 80.54 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801870, upper bound: 0.0801931
time: 37.20 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 117.82 seconds
NS_A1, status: Status.VERIFIED, split count: 1, time: 117.82
Output dim: 6, lower bound: -0.0797637, upper bound: 0.0799331
NS_A2, status: Status.UNKNOWN, split count: 1, time: 117.82
Output dim: 6, lower bound: -0.0801870, upper bound: 0.0801931

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.7018137, -3.1332502, -4.7018461, -3.1332505, -1.4616566, 1.4599485
1: -5.4192057, -3.1153355, -5.4192514, -3.1153345, -1.5820851, 1.5841951
2: -0.5015142, 0.0362498, -0.5015149, 0.0362719, -0.5325240, 0.5319521
3: -0.7205938, -0.1012130, -0.7205938, -0.1012127, -0.5054020, 0.5066031
4: -0.8135078, 0.1667680, -0.8135085, 0.1667686, -0.8534665, 0.8530289
5: -1.1506344, -0.5263911, -1.1506343, -0.5263705, -0.3982573, 0.3964950
6: 0.3552686, 0.5821691, 0.3552564, 0.5821691, -0.1224605, 0.1256009
7: -1.7589703, -0.4482925, -1.7589704, -0.4482720, -1.2073768, 1.1998541
8: -5.8517017, -3.9170167, -5.8517036, -3.9170165, -1.4380271, 1.4396284
9: -4.5359764, -2.7854781, -4.5359945, -2.7854774, -1.3312356, 1.3317360

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 334
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 268
type: B, layer: 1, pos: 3026
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 3035
type: B, layer: 1, pos: 3485
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 3185
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2703
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3408
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 3107
type: B, layer: 1, pos: 2705
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 2664
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 3261
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2546
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2415
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3293
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 3106
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 2412
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 3009
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 3215
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2308
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 2865
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2447
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 2458
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 2478
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 3309
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 2820
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 3294
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 3315
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3120
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 2683
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 3424
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2256
type: B, layer: 1, pos: 2490
type: B, layer: 1, pos: 469
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 2259
type: B, layer: 1, pos: 3017
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 2900
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 2899
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2791
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 2898
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2874
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 2855
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2309
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2910
type: B, layer: 1, pos: 2912
type: B, layer: 1, pos: 2913
type: B, layer: 1, pos: 2914
type: B, layer: 1, pos: 2918
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3135
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3194
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3370

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 334

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0800463, upper bound: 0.0798559
time: 33.05 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801697, upper bound: 0.0801706
time: 82.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 121.39 seconds
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 121.39
Output dim: 6, lower bound: -0.0800463, upper bound: 0.0798559
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 121.39
Output dim: 6, lower bound: -0.0801697, upper bound: 0.0801706

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4.6996822, -3.1332698, -4.6998110, -3.1326363, -1.4600434, 1.4605813
1: -5.4177504, -3.1153727, -5.4188671, -3.1149783, -1.5809146, 1.5836693
2: -0.5014184, 0.0362390, -0.5101156, 0.0362577, -0.5320759, 0.5403872
3: -0.7205830, -0.1022612, -0.7276649, -0.1020697, -0.5050135, 0.5091166
4: -0.8133993, 0.1667668, -0.8139474, 0.1704991, -0.8570555, 0.8528090
5: -1.1506236, -0.5270798, -1.1661087, -0.5269835, -0.3970602, 0.4061928
6: 0.3556585, 0.5821669, 0.3554054, 0.5854008, -0.1248228, 0.1250289
7: -1.7589505, -0.4487343, -1.7775078, -0.4396557, -1.2151821, 1.2181308
8: -5.8499203, -3.9170465, -5.8517742, -3.9165249, -1.4364420, 1.4410900
9: -4.5359683, -2.7855484, -4.5365739, -2.7855253, -1.3311807, 1.3322875

Time for backsubstitution: 6.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 273
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 304
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 268
type: A, layer: 1, pos: 3026
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 334
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 3246
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 2528
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 3035
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 3199
type: A, layer: 1, pos: 3111
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 3185
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3093
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2703
type: A, layer: 1, pos: 2290
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2704
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3408
type: A, layer: 1, pos: 3514
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2629
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 3107
type: A, layer: 1, pos: 2705
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 2664
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 3261
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 3060
type: A, layer: 1, pos: 2289
type: A, layer: 1, pos: 110
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2546
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2370
type: A, layer: 1, pos: 2415
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 2258
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 3293
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 2385
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 3106
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 3105
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2051
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 2412
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 3009
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 2508
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 3215
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2409
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2371
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2438
type: A, layer: 1, pos: 2308
type: A, layer: 1, pos: 3228
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2865
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 2413
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 2447
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3155
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 2548
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2674
type: A, layer: 1, pos: 3061
type: A, layer: 1, pos: 3300
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3049
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 2458
type: A, layer: 1, pos: 2478
type: A, layer: 1, pos: 2983
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 2607
type: A, layer: 1, pos: 3309
type: A, layer: 1, pos: 2411
type: A, layer: 1, pos: 2296
type: A, layer: 1, pos: 2820
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 2835
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 3294
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 2593
type: A, layer: 1, pos: 2050
type: A, layer: 1, pos: 2747
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 2659
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 3315
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 3121
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 3120
type: A, layer: 1, pos: 2683
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2505
type: A, layer: 1, pos: 3424
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2872
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 2256
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 469
type: A, layer: 1, pos: 2657
type: A, layer: 1, pos: 2130
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 2608
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2259
type: A, layer: 1, pos: 3017
type: A, layer: 1, pos: 2255
type: A, layer: 1, pos: 2900
type: A, layer: 1, pos: 3041
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 2928
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 2899
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2536
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2791
type: A, layer: 1, pos: 2938
type: A, layer: 1, pos: 2100
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2898
type: A, layer: 1, pos: 2101
type: A, layer: 1, pos: 2367
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 2254
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 2310
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2934
type: A, layer: 1, pos: 2253
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 2985
type: A, layer: 1, pos: 2855
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 2476
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2986
type: A, layer: 1, pos: 2521
type: A, layer: 1, pos: 2592
type: A, layer: 1, pos: 2927
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 3101
type: A, layer: 1, pos: 2251
type: A, layer: 1, pos: 2936
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 2144
type: A, layer: 1, pos: 2309
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2462
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2564
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2687
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2910
type: A, layer: 1, pos: 2912
type: A, layer: 1, pos: 2913
type: A, layer: 1, pos: 2914
type: A, layer: 1, pos: 2918
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 3014
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3135
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3138
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3140
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3149
type: A, layer: 1, pos: 3194
type: A, layer: 1, pos: 3269
type: A, layer: 1, pos: 3370

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 273

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0800007, upper bound: 0.0801687
time: 204.34 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801688, upper bound: 0.0801723
time: 78.87 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 289.61 seconds
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 289.61
Output dim: 6, lower bound: -0.0800007, upper bound: 0.0801687
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 289.61
Output dim: 6, lower bound: -0.0801688, upper bound: 0.0801723

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.6989908, -3.1332729, -4.6993856, -3.1326382, -1.4584920, 1.4592552
1: -5.4176903, -3.1154962, -5.4188309, -3.1150527, -1.5807972, 1.5835481
2: -0.5010123, 0.0362390, -0.5098724, 0.0362577, -0.5316895, 0.5401463
3: -0.7205658, -0.1024647, -0.7276540, -0.1021963, -0.5048281, 0.5088476
4: -0.8132411, 0.1667650, -0.8138497, 0.1704981, -0.8563464, 0.8520626
5: -1.1506212, -0.5273652, -1.1661071, -0.5271587, -0.3968775, 0.4058955
6: 0.3560999, 0.5821638, 0.3556682, 0.5853990, -0.1243510, 0.1247394
7: -1.7589000, -0.4490316, -1.7774771, -0.4398311, -1.2148013, 1.2176363
8: -5.8498430, -3.9173183, -5.8517289, -3.9166961, -1.4362867, 1.4409122
9: -4.5359373, -2.7864239, -4.5365562, -2.7860477, -1.3304880, 1.3312259

Time for backsubstitution: 6.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 268
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3026
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 3485
type: B, layer: 1, pos: 3035
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 3185
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2703
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 3408
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 3107
type: B, layer: 1, pos: 2705
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 2664
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 3261
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 2546
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2415
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3293
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 3106
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2412
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 3009
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 3215
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2308
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2865
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2447
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 2820
type: B, layer: 1, pos: 2478
type: B, layer: 1, pos: 2458
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 3309
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 3294
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 3315
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3120
type: B, layer: 1, pos: 2683
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 3424
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2256
type: B, layer: 1, pos: 2490
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 469
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 2259
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 2900
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 3017
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2899
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2791
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 2898
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2874
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 2855
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2309
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2910
type: B, layer: 1, pos: 2912
type: B, layer: 1, pos: 2913
type: B, layer: 1, pos: 2914
type: B, layer: 1, pos: 2918
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3135
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3194
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3370

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2573

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0799050, upper bound: 0.0800820
time: 63.68 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0799042, upper bound: 0.0800786
time: 193.07 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.7007008, -3.1307383, -4.6985493, -3.1326532, -1.4673138, 1.4632007
1: -5.4191399, -3.1153574, -5.4188585, -3.1150289, -1.5818462, 1.5836136
2: -0.5011383, 0.0403643, -0.5087101, 0.0362577, -0.5332725, 0.5430738
3: -0.7231388, -0.1013801, -0.7276394, -0.1020886, -0.5075127, 0.5100244
4: -0.8157846, 0.1677201, -0.8139404, 0.1704890, -0.8642291, 0.8525171
5: -1.1538966, -0.5270429, -1.1661072, -0.5270052, -0.4001864, 0.4061061
6: 0.3556460, 0.5866946, 0.3554171, 0.5853992, -0.1245796, 0.1294144
7: -1.7594767, -0.4478348, -1.7770227, -0.4396691, -1.2157419, 1.2199557
8: -5.8530869, -3.9169841, -5.8517647, -3.9165773, -1.4377911, 1.4411029
9: -4.5465670, -2.7855735, -4.5365515, -2.7855654, -1.3414793, 1.3324414

Time for backsubstitution: 6.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 304
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 268
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3026
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 3246
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 2528
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 3485
type: B, layer: 1, pos: 3035
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 273
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 3199
type: B, layer: 1, pos: 3111
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 3185
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3093
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2703
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2290
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 2704
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 3514
type: B, layer: 1, pos: 3408
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 2629
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 3107
type: B, layer: 1, pos: 2705
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 2664
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 3261
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2289
type: B, layer: 1, pos: 3060
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 110
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 2546
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2415
type: B, layer: 1, pos: 2370
type: B, layer: 1, pos: 2258
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 3293
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2385
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 3106
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 3105
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 2051
type: B, layer: 1, pos: 2412
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2508
type: B, layer: 1, pos: 3009
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 3215
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2409
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2371
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 2438
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3228
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2308
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2865
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2413
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 2447
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 2674
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 3061
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 3049
type: B, layer: 1, pos: 2607
type: B, layer: 1, pos: 2820
type: B, layer: 1, pos: 2478
type: B, layer: 1, pos: 2458
type: B, layer: 1, pos: 2983
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2411
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 3309
type: B, layer: 1, pos: 2835
type: B, layer: 1, pos: 2296
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 3294
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2050
type: B, layer: 1, pos: 2747
type: B, layer: 1, pos: 2593
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 2659
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 3315
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 3121
type: B, layer: 1, pos: 3120
type: B, layer: 1, pos: 2683
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 3424
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 2872
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 2256
type: B, layer: 1, pos: 2490
type: B, layer: 1, pos: 2657
type: B, layer: 1, pos: 469
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 2130
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2608
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 2259
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 2900
type: B, layer: 1, pos: 2255
type: B, layer: 1, pos: 3017
type: B, layer: 1, pos: 3041
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 2928
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 2899
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2536
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 2791
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2100
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 2898
type: B, layer: 1, pos: 2101
type: B, layer: 1, pos: 2874
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2367
type: B, layer: 1, pos: 2254
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 2310
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 2934
type: B, layer: 1, pos: 2253
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 2855
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 2985
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2521
type: B, layer: 1, pos: 2986
type: B, layer: 1, pos: 3101
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2927
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 2476
type: B, layer: 1, pos: 2592
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 2251
type: B, layer: 1, pos: 2936
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2309
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2564
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2687
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2910
type: B, layer: 1, pos: 2912
type: B, layer: 1, pos: 2913
type: B, layer: 1, pos: 2914
type: B, layer: 1, pos: 2918
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 3014
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3135
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3138
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3140
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3149
type: B, layer: 1, pos: 3194
type: B, layer: 1, pos: 3269
type: B, layer: 1, pos: 3370

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2573

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0800750, upper bound: 0.0800854
time: 47.69 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0800758, upper bound: 0.0800800
time: 51.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 105.97 seconds
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 105.97
Output dim: 6, lower bound: -0.0799050, upper bound: 0.0800820
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 105.97
Output dim: 6, lower bound: -0.0799042, upper bound: 0.0800786
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 105.97
Output dim: 6, lower bound: -0.0800750, upper bound: 0.0800854
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 105.97
Output dim: 6, lower bound: -0.0800758, upper bound: 0.0800800

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 349.96 + 897.96 = 1247.92 seconds
