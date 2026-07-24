## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 12)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.2577547872


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4171969, 0.4171970)
1: (-0.8265239, 0.9890501, -0.8265239, 0.9890501, -1.2514708, 1.2514708)
2: (-2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6294950, 1.6294951)
3: (-7.1971846, -3.3857136, -7.1971846, -3.3857136, -1.4862355, 1.4862354)
4: (-6.0309587, -2.7082672, -6.0309587, -2.7082672, -1.6490123, 1.6490123)
5: (-7.5052495, -3.6067972, -7.5052495, -3.6067972, -1.6530236, 1.6530237)
6: (-10.5238905, -5.7908278, -10.5238905, -5.7908278, -1.4879827, 1.4879827)
7: (-6.7266016, -1.9406551, -6.7266016, -1.9406551, -3.1986961, 3.1986961)
8: (-1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0515521, 1.0515522)
9: (-1.4500365, 0.4697701, -1.4500365, 0.4697701, -1.6120032, 1.6120033)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.47 + 86.94 = 95.41 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.2580120, upper bound: 0.2580108

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2325
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 3374
type: DSZ, layer: 1, pos: 570
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3006
type: DSZ, layer: 1, pos: 296
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 562
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2404
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3306
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2467
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2724
type: DSZ, layer: 1, pos: 3034
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2960
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 894
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 293
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3127
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2412
type: DSZ, layer: 1, pos: 2548
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2980
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2531
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 792
type: DSZ, layer: 1, pos: 2663
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 361
type: DSZ, layer: 1, pos: 3285
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 3087
type: DSZ, layer: 1, pos: 3550
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 3277
type: DSZ, layer: 1, pos: 3447
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2397
type: DSZ, layer: 1, pos: 3316
type: DSZ, layer: 1, pos: 2516
type: DSZ, layer: 1, pos: 2723
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 3539
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2315
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 746
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 623
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3088
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 3582
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 281
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 608
type: DSZ, layer: 1, pos: 2742
type: DSZ, layer: 1, pos: 2562
type: DSZ, layer: 1, pos: 3299
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 3270
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 3533
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 3030
type: DSZ, layer: 1, pos: 2316
type: DSZ, layer: 1, pos: 3340
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2378
type: DSZ, layer: 1, pos: 3337
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 3487
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 449
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 3254
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3470
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2338
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2272
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 599
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2740
type: DSZ, layer: 1, pos: 3325
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3071
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2092
type: DSZ, layer: 1, pos: 672
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2146
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 3001
type: DSZ, layer: 1, pos: 3243
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2588
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 612
type: DSZ, layer: 1, pos: 2999
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 2274
type: DSZ, layer: 1, pos: 2401
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 3237
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2987
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2725
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2722
type: DSZ, layer: 1, pos: 3223
type: DSZ, layer: 1, pos: 610
type: DSZ, layer: 1, pos: 294
type: DSZ, layer: 1, pos: 554
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2988
type: DSZ, layer: 1, pos: 2684
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 593
type: DSZ, layer: 1, pos: 3563
type: DSZ, layer: 1, pos: 3119
type: DSZ, layer: 1, pos: 3566
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 725
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2429
type: DSZ, layer: 1, pos: 3264
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2967
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2741
type: DSZ, layer: 1, pos: 2739
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2430
type: DSZ, layer: 1, pos: 2153
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 3473
type: DSZ, layer: 1, pos: 3141
type: DSZ, layer: 1, pos: 3291
type: DSZ, layer: 1, pos: 2558
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2759
type: DSZ, layer: 1, pos: 611
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2273
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2325

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2568294, upper bound: 0.2568464
time: 346.25 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2568484, upper bound: 0.2568288
time: 632.05 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 978.32 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 978.32
Output dim: 1, lower bound: -0.2568294, upper bound: 0.2568464
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 978.32
Output dim: 1, lower bound: -0.2568484, upper bound: 0.2568288

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 95.41 + 978.32 = 1073.73 seconds
