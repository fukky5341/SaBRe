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
execution time: IAR + RelationalAnalysis = 7.94 + 352.11 = 360.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0801936, upper bound: 0.0801986

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2308
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 3309
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2912
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 3120
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 3194
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2462

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2514

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801850, upper bound: 0.0801859
time: 217.38 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801813, upper bound: 0.0801873
time: 262.06 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 479.45 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 479.45
Output dim: 6, lower bound: -0.0801850, upper bound: 0.0801859
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 479.45
Output dim: 6, lower bound: -0.0801813, upper bound: 0.0801873

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.7020321, -3.1332498, -4.7020321, -3.1332498, -1.4604194, 1.4604208
1: -5.4196887, -3.1153350, -5.4196887, -3.1153350, -1.5848356, 1.5848358
2: -0.5015193, 0.0363976, -0.5015193, 0.0363976, -0.5326592, 0.5326617
3: -0.7205948, -0.1012112, -0.7205948, -0.1012112, -0.5057970, 0.5057963
4: -0.8135126, 0.1667712, -0.8135126, 0.1667712, -0.8530917, 0.8530923
5: -1.1506350, -0.5262530, -1.1506350, -0.5262530, -0.3987786, 0.3987790
6: 0.3551826, 0.5821692, 0.3551826, 0.5821692, -0.1256490, 0.1256488
7: -1.7589732, -0.4479041, -1.7589732, -0.4479041, -1.2074827, 1.2074842
8: -5.8517289, -3.9170151, -5.8517289, -3.9170151, -1.4400725, 1.4400725
9: -4.5360956, -2.7854774, -4.5360956, -2.7854774, -1.3318336, 1.3318326

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 3120
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3309
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2308
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2912
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 3194
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2430

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2674

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801809, upper bound: 0.0801822
time: 370.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801779, upper bound: 0.0801809
time: 128.65 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.7020321, -3.1332498, -4.7020321, -3.1332498, -1.4604206, 1.4604194
1: -5.4196887, -3.1153350, -5.4196887, -3.1153350, -1.5848356, 1.5848358
2: -0.5015193, 0.0363976, -0.5015193, 0.0363976, -0.5326617, 0.5326592
3: -0.7205948, -0.1012112, -0.7205948, -0.1012112, -0.5057963, 0.5057970
4: -0.8135126, 0.1667712, -0.8135126, 0.1667712, -0.8530924, 0.8530917
5: -1.1506350, -0.5262530, -1.1506350, -0.5262530, -0.3987789, 0.3987786
6: 0.3551826, 0.5821692, 0.3551826, 0.5821692, -0.1256488, 0.1256490
7: -1.7589732, -0.4479041, -1.7589732, -0.4479041, -1.2074842, 1.2074828
8: -5.8517289, -3.9170151, -5.8517289, -3.9170151, -1.4400724, 1.4400725
9: -4.5360956, -2.7854774, -4.5360956, -2.7854774, -1.3318325, 1.3318336

Time for backsubstitution: 5.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 3194
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 2308
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2912
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3309
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3120
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2385

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3098

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801687, upper bound: 0.0801815
time: 306.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801758, upper bound: 0.0801731
time: 133.12 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 445.10 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 445.10
Output dim: 6, lower bound: -0.0801809, upper bound: 0.0801822
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 445.10
Output dim: 6, lower bound: -0.0801779, upper bound: 0.0801809
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 445.10
Output dim: 6, lower bound: -0.0801687, upper bound: 0.0801815
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 445.10
Output dim: 6, lower bound: -0.0801758, upper bound: 0.0801731

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7020321, -3.1332498, -4.7020321, -3.1332498, -1.4601357, 1.4602185
1: -5.4196887, -3.1153350, -5.4196887, -3.1153350, -1.5835699, 1.5839099
2: -0.5015193, 0.0363976, -0.5015193, 0.0363976, -0.5326489, 0.5326533
3: -0.7205948, -0.1012112, -0.7205948, -0.1012112, -0.5057969, 0.5057962
4: -0.8135126, 0.1667712, -0.8135126, 0.1667712, -0.8530834, 0.8530830
5: -1.1506350, -0.5262530, -1.1506350, -0.5262530, -0.3987586, 0.3987541
6: 0.3551826, 0.5821692, 0.3551826, 0.5821692, -0.1256487, 0.1256485
7: -1.7589732, -0.4479041, -1.7589732, -0.4479041, -1.2074708, 1.2074680
8: -5.8517289, -3.9170151, -5.8517289, -3.9170151, -1.4389000, 1.4392645
9: -4.5360956, -2.7854774, -4.5360956, -2.7854774, -1.3305420, 1.3309066

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3309
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2308
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3194
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2912
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 3120
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2593

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2050

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801681, upper bound: 0.0801784
time: 35.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801697, upper bound: 0.0801672
time: 29.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7020321, -3.1332498, -4.7020321, -3.1332498, -1.4602172, 1.4601370
1: -5.4196887, -3.1153350, -5.4196887, -3.1153350, -1.5839099, 1.5835699
2: -0.5015193, 0.0363976, -0.5015193, 0.0363976, -0.5326508, 0.5326514
3: -0.7205948, -0.1012112, -0.7205948, -0.1012112, -0.5057969, 0.5057962
4: -0.8135126, 0.1667712, -0.8135126, 0.1667712, -0.8530824, 0.8530841
5: -1.1506350, -0.5262530, -1.1506350, -0.5262530, -0.3987538, 0.3987590
6: 0.3551826, 0.5821692, 0.3551826, 0.5821692, -0.1256487, 0.1256485
7: -1.7589732, -0.4479041, -1.7589732, -0.4479041, -1.2074666, 1.2074723
8: -5.8517289, -3.9170151, -5.8517289, -3.9170151, -1.4392647, 1.4388998
9: -4.5360956, -2.7854774, -4.5360956, -2.7854774, -1.3309076, 1.3305409

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2912
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2308
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 3309
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3194
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3120
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2289

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2564

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801828, upper bound: 0.0801847
time: 112.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801828, upper bound: 0.0801835
time: 290.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7020321, -3.1332498, -4.7020321, -3.1332498, -1.4543120, 1.4539678
1: -5.4196887, -3.1153350, -5.4196887, -3.1153350, -1.5741975, 1.5735779
2: -0.5015193, 0.0363976, -0.5015193, 0.0363976, -0.5325794, 0.5325635
3: -0.7205948, -0.1012112, -0.7205948, -0.1012112, -0.5051569, 0.5052006
4: -0.8135126, 0.1667712, -0.8135126, 0.1667712, -0.8530151, 0.8530188
5: -1.1506350, -0.5262530, -1.1506350, -0.5262530, -0.3976794, 0.3977229
6: 0.3551826, 0.5821692, 0.3551826, 0.5821692, -0.1256303, 0.1256313
7: -1.7589732, -0.4479041, -1.7589732, -0.4479041, -1.2073859, 1.2073863
8: -5.8517289, -3.9170151, -5.8517289, -3.9170151, -1.4314065, 1.4308786
9: -4.5360956, -2.7854774, -4.5360956, -2.7854774, -1.3238552, 1.3233955

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 3194
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 3120
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2308
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2912
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3309
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3485

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2273

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801598, upper bound: 0.0801755
time: 213.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801622, upper bound: 0.0801642
time: 384.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7020321, -3.1332498, -4.7020321, -3.1332498, -1.4539692, 1.4543107
1: -5.4196887, -3.1153350, -5.4196887, -3.1153350, -1.5735779, 1.5741973
2: -0.5015193, 0.0363976, -0.5015193, 0.0363976, -0.5325660, 0.5325770
3: -0.7205948, -0.1012112, -0.7205948, -0.1012112, -0.5052000, 0.5051575
4: -0.8135126, 0.1667712, -0.8135126, 0.1667712, -0.8530195, 0.8530144
5: -1.1506350, -0.5262530, -1.1506350, -0.5262530, -0.3977232, 0.3976791
6: 0.3551826, 0.5821692, 0.3551826, 0.5821692, -0.1256311, 0.1256305
7: -1.7589732, -0.4479041, -1.7589732, -0.4479041, -1.2073878, 1.2073843
8: -5.8517289, -3.9170151, -5.8517289, -3.9170151, -1.4308783, 1.4314065
9: -4.5360956, -2.7854774, -4.5360956, -2.7854774, -1.3233943, 1.3238564

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 3120
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2674
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2912
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3309
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2308
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 3194
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3101

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801741, upper bound: 0.0801734
time: 251.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801734, upper bound: 0.0801738
time: 65.74 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 323.19 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 323.19
Output dim: 6, lower bound: -0.0801681, upper bound: 0.0801784
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 323.19
Output dim: 6, lower bound: -0.0801697, upper bound: 0.0801672
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 323.19
Output dim: 6, lower bound: -0.0801828, upper bound: 0.0801847
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 323.19
Output dim: 6, lower bound: -0.0801828, upper bound: 0.0801835
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 323.19
Output dim: 6, lower bound: -0.0801598, upper bound: 0.0801755
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 323.19
Output dim: 6, lower bound: -0.0801622, upper bound: 0.0801642
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 323.19
Output dim: 6, lower bound: -0.0801741, upper bound: 0.0801734
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 323.19
Output dim: 6, lower bound: -0.0801734, upper bound: 0.0801738

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7020321, -3.1332498, -4.7020321, -3.1332498, -1.4601350, 1.4602180
1: -5.4196887, -3.1153350, -5.4196887, -3.1153350, -1.5835696, 1.5839097
2: -0.5015193, 0.0363976, -0.5015193, 0.0363976, -0.5326488, 0.5326532
3: -0.7205948, -0.1012112, -0.7205948, -0.1012112, -0.5057968, 0.5057962
4: -0.8135126, 0.1667712, -0.8135126, 0.1667712, -0.8530831, 0.8530828
5: -1.1506350, -0.5262530, -1.1506350, -0.5262530, -0.3987586, 0.3987541
6: 0.3551826, 0.5821692, 0.3551826, 0.5821692, -0.1256486, 0.1256484
7: -1.7589732, -0.4479041, -1.7589732, -0.4479041, -1.2074705, 1.2074679
8: -5.8517289, -3.9170151, -5.8517289, -3.9170151, -1.4388993, 1.4392641
9: -4.5360956, -2.7854774, -4.5360956, -2.7854774, -1.3305417, 1.3309063

Time for backsubstitution: 5.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3194
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 3309
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 2308
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 3120
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2912
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 3106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2258

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801621, upper bound: 0.0801653
time: 144.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801612, upper bound: 0.0801746
time: 21.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7020321, -3.1332498, -4.7020321, -3.1332498, -1.4601350, 1.4602180
1: -5.4196887, -3.1153350, -5.4196887, -3.1153350, -1.5835696, 1.5839097
2: -0.5015193, 0.0363976, -0.5015193, 0.0363976, -0.5326488, 0.5326532
3: -0.7205948, -0.1012112, -0.7205948, -0.1012112, -0.5057968, 0.5057962
4: -0.8135126, 0.1667712, -0.8135126, 0.1667712, -0.8530831, 0.8530828
5: -1.1506350, -0.5262530, -1.1506350, -0.5262530, -0.3987586, 0.3987541
6: 0.3551826, 0.5821692, 0.3551826, 0.5821692, -0.1256486, 0.1256484
7: -1.7589732, -0.4479041, -1.7589732, -0.4479041, -1.2074705, 1.2074679
8: -5.8517289, -3.9170151, -5.8517289, -3.9170151, -1.4388993, 1.4392641
9: -4.5360956, -2.7854774, -4.5360956, -2.7854774, -1.3305417, 1.3309063

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2593
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 334
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3370
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2256
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 689
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3060
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3093
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2290
type: DSZ, layer: 1, pos: 3514
type: DSZ, layer: 1, pos: 2900
type: DSZ, layer: 1, pos: 3246
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 2918
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2983
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2704
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 3061
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 2409
type: DSZ, layer: 1, pos: 590
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 3026
type: DSZ, layer: 1, pos: 2413
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2608
type: DSZ, layer: 1, pos: 2253
type: DSZ, layer: 1, pos: 2478
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2928
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2835
type: DSZ, layer: 1, pos: 2659
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2254
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2258
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2938
type: DSZ, layer: 1, pos: 3014
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2310
type: DSZ, layer: 1, pos: 3135
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2385
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2855
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2564
type: DSZ, layer: 1, pos: 2747
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2874
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 3199
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2528
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2415
type: DSZ, layer: 1, pos: 2683
type: DSZ, layer: 1, pos: 3009
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2910
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2934
type: DSZ, layer: 1, pos: 2296
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2912
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2255
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2865
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2898
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 3111
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3035
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2499
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2289
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2508
type: DSZ, layer: 1, pos: 3149
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 3017
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2607
type: DSZ, layer: 1, pos: 3101
type: DSZ, layer: 1, pos: 2899
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2367
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2820
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 3049
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3107
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 304
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 3315
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3105
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2913
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 3120
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2521
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3194
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2259
type: DSZ, layer: 1, pos: 2629
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 797
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 273
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2927
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3185
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3041
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2309
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2308
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2705
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2703
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 268
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2872
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3293
type: DSZ, layer: 1, pos: 2914
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 3261
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3309
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2476
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3228
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2251
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2936
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2471

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 368

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0801324, upper bound: 0.0800902
time: 79.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0800933, upper bound: 0.0801251
time: 290.69 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 375.86 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 375.86
Output dim: 6, lower bound: -0.0801621, upper bound: 0.0801653
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 375.86
Output dim: 6, lower bound: -0.0801612, upper bound: 0.0801746
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 375.86
Output dim: 6, lower bound: -0.0801324, upper bound: 0.0800902
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 375.86
Output dim: 6, lower bound: -0.0800933, upper bound: 0.0801251
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 375.86
Output dim: 6, lower bound: -0.0801828, upper bound: 0.0801847
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 375.86
Output dim: 6, lower bound: -0.0801828, upper bound: 0.0801835
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 375.86
Output dim: 6, lower bound: -0.0801598, upper bound: 0.0801755
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 375.86
Output dim: 6, lower bound: -0.0801622, upper bound: 0.0801642
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 375.86
Output dim: 6, lower bound: -0.0801741, upper bound: 0.0801734
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 375.86
Output dim: 6, lower bound: -0.0801734, upper bound: 0.0801738

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 360.05 + 3384.62 = 3744.67 seconds
