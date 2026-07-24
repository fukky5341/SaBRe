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
execution time: IAR + RelationalAnalysis = 7.80 + 347.32 = 355.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0801936, upper bound: 0.0801986

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2308
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2912
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3120
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3194
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3309
type: DSZ, layer: 1, pos: 3370

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3023

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0800929, upper bound: 0.0801130
time: 32.08 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0801081, upper bound: 0.0800994
time: 49.17 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 81.33 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 81.33
Output dim: 6, lower bound: -0.0800929, upper bound: 0.0801130
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 81.33
Output dim: 6, lower bound: -0.0801081, upper bound: 0.0800994

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 355.12 + 81.33 = 436.46 seconds
