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
execution time: IAR + RelationalAnalysis = 9.68 + 87.80 = 97.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.2580120, upper bound: 0.2580108

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 293
type: A, layer: 1, pos: 422
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 3487
type: A, layer: 1, pos: 2967
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 361
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2447
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 296
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 2742
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2741
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 281
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 3563
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 2740
type: A, layer: 1, pos: 2548
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2743
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 3325
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 3473
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 3270
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2725
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 2739
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2738
type: A, layer: 1, pos: 2412
type: A, layer: 1, pos: 3243
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 2723
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2722
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 3533
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 2737
type: A, layer: 1, pos: 3343
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 3223
type: A, layer: 1, pos: 3237
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 3582
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2467
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2579
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 2759
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3119
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3539
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 294

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2554287, upper bound: 0.2563413
time: 311.11 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2580073, upper bound: 0.2580061
time: 191.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 502.95 seconds
NS_A1, status: Status.VERIFIED, split count: 1, time: 502.95
Output dim: 1, lower bound: -0.2554287, upper bound: 0.2563413
NS_A2, status: Status.UNKNOWN, split count: 1, time: 502.95
Output dim: 1, lower bound: -0.2580073, upper bound: 0.2580061

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.7613775, 0.3492614, -0.7613798, 0.3492641, -0.4171712, 0.4060650
1: -0.8265101, 0.9890465, -0.8265124, 0.9890472, -1.2366221, 1.2514319
2: -2.7041450, -0.1663677, -2.7041476, -0.1663078, -1.6291885, 1.6282234
3: -7.1971817, -3.3863957, -7.1971817, -3.3862906, -1.4859738, 1.4867691
4: -6.0309362, -2.7083230, -6.0309396, -2.7083144, -1.6489787, 1.6426271
5: -7.5052214, -3.6084094, -7.5052271, -3.6081889, -1.6524956, 1.6517986
6: -10.5238705, -5.7908487, -10.5238733, -5.7908459, -1.4840108, 1.4879478
7: -6.7265449, -1.9412842, -6.7265530, -1.9411875, -3.1982422, 3.1989465
8: -1.2038989, 0.5512741, -1.2038991, 0.5512741, -1.0515143, 1.0451136
9: -1.4500250, 0.4696424, -1.4500263, 0.4696620, -1.6122073, 1.6115806

Time for backsubstitution: 7.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 293
type: B, layer: 1, pos: 422
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 3487
type: B, layer: 1, pos: 2967
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 361
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2447
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 296
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2742
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2741
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2325
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 281
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 3563
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2743
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 3325
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 3473
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 3270
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2725
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 2739
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2738
type: B, layer: 1, pos: 2412
type: B, layer: 1, pos: 3243
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 2723
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2722
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 3533
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 2737
type: B, layer: 1, pos: 3343
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 3223
type: B, layer: 1, pos: 3237
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 3582
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 449
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2467
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 2759
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3119
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3254
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3539
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 293

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2578728, upper bound: 0.2567799
time: 46.53 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2580109, upper bound: 0.2580071
time: 69.48 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 123.94 seconds
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 123.94
Output dim: 1, lower bound: -0.2578728, upper bound: 0.2567799
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 123.94
Output dim: 1, lower bound: -0.2580109, upper bound: 0.2580071

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.7613197, 0.3483509, -0.7613021, 0.3480007, -0.4158054, 0.4050255
1: -0.8240811, 0.9888040, -0.8231850, 0.9887187, -1.2340465, 1.2479942
2: -2.7039757, -0.1664494, -2.7039130, -0.1664213, -1.6287539, 1.6278224
3: -7.1971388, -3.3864746, -7.1971235, -3.3863988, -1.4857063, 1.4865628
4: -6.0306668, -2.7092345, -6.0305676, -2.7095799, -1.6475759, 1.6414754
5: -7.5049882, -3.6084669, -7.5049067, -3.6082606, -1.6518836, 1.6512387
6: -10.5230532, -5.7910862, -10.5227394, -5.7911730, -1.4829836, 1.4866610
7: -6.7264547, -1.9413635, -6.7264304, -1.9412971, -3.1976538, 3.1984496
8: -1.2038376, 0.5509076, -1.2038176, 0.5507667, -1.0509186, 1.0446367
9: -1.4494486, 0.4694923, -1.4492266, 0.4694548, -1.6110542, 1.6103352

Time for backsubstitution: 7.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 422
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 3487
type: A, layer: 1, pos: 2967
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 361
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2447
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 296
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 293
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 2742
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2741
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 281
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 3563
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 2740
type: A, layer: 1, pos: 2548
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2743
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 3325
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 3473
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 3270
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2725
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 2739
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2738
type: A, layer: 1, pos: 2412
type: A, layer: 1, pos: 3243
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 2723
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2722
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 3533
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 2737
type: A, layer: 1, pos: 3343
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 3223
type: A, layer: 1, pos: 3237
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 3582
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2467
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2579
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 2759
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3119
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3539
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 422

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2574968, upper bound: 0.2567664
time: 240.35 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2578605, upper bound: 0.2567715
time: 32.75 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.7613146, 0.3491955, -0.7730955, 0.3492965, -0.4160337, 0.4176137
1: -0.8264537, 0.9890373, -0.8268543, 1.0214630, -1.2690063, 1.2508452
2: -2.7040720, -0.1705309, -2.7063127, -0.1712277, -1.6283121, 1.6296108
3: -7.1970887, -3.3892434, -7.1970162, -3.3891616, -1.4852556, 1.4856658
4: -6.0308609, -2.7083783, -6.0484715, -2.7077303, -1.6488757, 1.6614583
5: -7.5050087, -3.6154439, -7.5050945, -3.6168349, -1.6495339, 1.6552441
6: -10.5237160, -5.7909403, -10.5237045, -5.7596550, -1.5166433, 1.4871500
7: -6.7261438, -1.9484212, -6.7437963, -1.9499300, -3.1936462, 3.2154655
8: -1.2038579, 0.5511286, -1.2124596, 0.5520147, -1.0526764, 1.0521157
9: -1.4499525, 0.4692680, -1.4520165, 0.4693427, -1.6101369, 1.6144176

Time for backsubstitution: 7.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 422
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 3487
type: A, layer: 1, pos: 2967
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 361
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2518
type: A, layer: 1, pos: 3006
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2293
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2558
type: A, layer: 1, pos: 2588
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2988
type: A, layer: 1, pos: 2447
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 296
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 3034
type: A, layer: 1, pos: 2430
type: A, layer: 1, pos: 2671
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 2359
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 293
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 2341
type: A, layer: 1, pos: 2092
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3030
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2572
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 2316
type: A, layer: 1, pos: 2742
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 3550
type: A, layer: 1, pos: 2597
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 2378
type: A, layer: 1, pos: 2741
type: A, layer: 1, pos: 3001
type: A, layer: 1, pos: 2325
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 2146
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 2960
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 3544
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2395
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 281
type: A, layer: 1, pos: 2987
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2401
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 3563
type: A, layer: 1, pos: 3128
type: A, layer: 1, pos: 2740
type: A, layer: 1, pos: 2548
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2974
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 2274
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2272
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 2562
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2404
type: A, layer: 1, pos: 3264
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 2743
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2273
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 3325
type: A, layer: 1, pos: 3127
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 2315
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 3071
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 2153
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 3473
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3291
type: A, layer: 1, pos: 2663
type: A, layer: 1, pos: 3306
type: A, layer: 1, pos: 3337
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 3285
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 3316
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 3277
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 3118
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 3470
type: A, layer: 1, pos: 3270
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 3087
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 2725
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 3088
type: A, layer: 1, pos: 2739
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2738
type: A, layer: 1, pos: 2412
type: A, layer: 1, pos: 3243
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2383
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 2397
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 3447
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2677
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 2723
type: A, layer: 1, pos: 2078
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2722
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 3322
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 3566
type: A, layer: 1, pos: 2724
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 3533
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 2737
type: A, layer: 1, pos: 3343
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 3340
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 3223
type: A, layer: 1, pos: 3237
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 3582
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 449
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2429
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2467
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2579
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2699
type: A, layer: 1, pos: 2759
type: A, layer: 1, pos: 2984
type: A, layer: 1, pos: 2999
type: A, layer: 1, pos: 3119
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3141
type: A, layer: 1, pos: 3254
type: A, layer: 1, pos: 3299
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3539
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 422

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2576304, upper bound: 0.2579989
time: 37.68 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579998, upper bound: 0.2579989
time: 52.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 98.07 seconds
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 98.07
Output dim: 1, lower bound: -0.2574968, upper bound: 0.2567664
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 98.07
Output dim: 1, lower bound: -0.2578605, upper bound: 0.2567715
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 98.07
Output dim: 1, lower bound: -0.2576304, upper bound: 0.2579989
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 98.07
Output dim: 1, lower bound: -0.2579998, upper bound: 0.2579989

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.7613127, 0.3483478, -0.7612955, 0.3479979, -0.4157975, 0.4016183
1: -0.8240442, 0.9887974, -0.8231524, 0.9887125, -1.2258911, 1.2479506
2: -2.7039630, -0.1664604, -2.7039018, -0.1664311, -1.6287324, 1.6214682
3: -7.1971340, -3.3865161, -7.1971188, -3.3864365, -1.4856272, 1.4611863
4: -6.0306621, -2.7100112, -6.0305629, -2.7102618, -1.6499498, 1.6406636
5: -7.5049787, -3.6084862, -7.5048981, -3.6082783, -1.6517669, 1.6293988
6: -10.5230532, -5.7911797, -10.5227385, -5.7912550, -1.4828675, 1.4487252
7: -6.7263985, -1.9413662, -6.7263803, -1.9412991, -3.1930399, 3.1983800
8: -1.2038248, 0.5508962, -1.2038057, 0.5507568, -1.0497785, 1.0446122
9: -1.4494138, 0.4694866, -1.4491966, 0.4694498, -1.6102957, 1.6125433

Time for backsubstitution: 7.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 3487
type: B, layer: 1, pos: 2967
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 361
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2447
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 296
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2742
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2741
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2325
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 422
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 281
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 3563
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2743
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 3325
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 3473
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 3270
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2725
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 2739
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2738
type: B, layer: 1, pos: 2412
type: B, layer: 1, pos: 3243
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2723
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2722
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 3533
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 2737
type: B, layer: 1, pos: 3343
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 3223
type: B, layer: 1, pos: 3237
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 3582
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 449
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2467
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 2759
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3119
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3254
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3539
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2980

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2576510, upper bound: 0.2566078
time: 453.53 seconds

## Relational analysis of NS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 408

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2577376, upper bound: 0.2562381
time: 254.18 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2577367, upper bound: 0.2566441
time: 255.13 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.7591194, 0.3463213, -0.7727430, 0.3467932, -0.4114397, 0.4144288
1: -0.8168523, 0.9860348, -0.8187588, 1.0214002, -1.2598158, 1.2401805
2: -2.6979582, -0.1842738, -2.7037477, -0.1841087, -1.6080174, 1.6128522
3: -7.1906071, -3.4101319, -7.1965771, -3.4086313, -1.4585723, 1.4637263
4: -6.0300069, -2.7094274, -6.0467215, -2.7087545, -1.6434580, 1.6565464
5: -7.5009174, -3.6318994, -7.5047336, -3.6323953, -1.6277654, 1.6370702
6: -10.5148392, -5.8227549, -10.5236759, -5.7891717, -1.4769313, 1.4537697
7: -6.7219987, -1.9512115, -6.7391644, -1.9503840, -3.1891940, 3.2082281
8: -1.2006955, 0.5510275, -1.2095865, 0.5512130, -1.0484457, 1.0486820
9: -1.4423048, 0.4643219, -1.4456124, 0.4687644, -1.6019586, 1.6007133

Time for backsubstitution: 7.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 3487
type: B, layer: 1, pos: 2967
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 361
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2447
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 296
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2742
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2741
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2325
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 422
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 281
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 3563
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2743
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 3325
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 3473
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 3270
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2725
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 2739
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2738
type: B, layer: 1, pos: 2412
type: B, layer: 1, pos: 3243
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2723
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2722
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 3533
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 2737
type: B, layer: 1, pos: 3343
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 3223
type: B, layer: 1, pos: 3237
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 3582
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 449
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2467
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 2759
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3119
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3254
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3539
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2980

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2574295, upper bound: 0.2578426
time: 425.30 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2574250, upper bound: 0.2577883
time: 309.71 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.7613078, 0.3491921, -0.7730893, 0.3492937, -0.4160260, 0.4142064
1: -0.8264170, 0.9890302, -0.8268222, 1.0214572, -1.2608517, 1.2508022
2: -2.7040598, -0.1705415, -2.7063017, -0.1712366, -1.6282908, 1.6232566
3: -7.1970844, -3.3892853, -7.1970115, -3.3891995, -1.4851762, 1.4602895
4: -6.0308552, -2.7091556, -6.0484662, -2.7084122, -1.6512494, 1.6606462
5: -7.5050001, -3.6154647, -7.5050859, -3.6168532, -1.6494174, 1.6334074
6: -10.5237169, -5.7910337, -10.5237045, -5.7597361, -1.5165296, 1.4492143
7: -6.7260876, -1.9484236, -6.7437463, -1.9499319, -3.1890326, 3.2153964
8: -1.2038441, 0.5511172, -1.2124480, 0.5520048, -1.0515426, 1.0520914
9: -1.4499179, 0.4692618, -1.4519863, 0.4693375, -1.6093785, 1.6166248

Time for backsubstitution: 7.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 3487
type: B, layer: 1, pos: 2967
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2080
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 361
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2518
type: B, layer: 1, pos: 3006
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2293
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2558
type: B, layer: 1, pos: 2588
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2988
type: B, layer: 1, pos: 2447
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 296
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 3034
type: B, layer: 1, pos: 2430
type: B, layer: 1, pos: 2671
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 2359
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 2341
type: B, layer: 1, pos: 2092
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3030
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2572
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 2316
type: B, layer: 1, pos: 2742
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 3550
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 2378
type: B, layer: 1, pos: 2741
type: B, layer: 1, pos: 3001
type: B, layer: 1, pos: 2325
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 2146
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 422
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 2960
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 3544
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2395
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 281
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2987
type: B, layer: 1, pos: 2401
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 3563
type: B, layer: 1, pos: 3128
type: B, layer: 1, pos: 2740
type: B, layer: 1, pos: 2548
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2974
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 2274
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2272
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 2562
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 2404
type: B, layer: 1, pos: 3264
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 2743
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2273
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 3325
type: B, layer: 1, pos: 3127
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 2214
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 2315
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 3071
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 2153
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 3473
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 3291
type: B, layer: 1, pos: 2663
type: B, layer: 1, pos: 3306
type: B, layer: 1, pos: 3337
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 3285
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 3316
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 3277
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 3118
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 3470
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 3270
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 3087
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2725
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 3088
type: B, layer: 1, pos: 2739
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2738
type: B, layer: 1, pos: 2412
type: B, layer: 1, pos: 3243
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2383
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2397
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 3447
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2677
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 2078
type: B, layer: 1, pos: 2723
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2722
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 3322
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 3566
type: B, layer: 1, pos: 2724
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 3533
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 2737
type: B, layer: 1, pos: 3343
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3340
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 3223
type: B, layer: 1, pos: 3237
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 3582
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 449
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2429
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2467
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2579
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2699
type: B, layer: 1, pos: 2759
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 2999
type: B, layer: 1, pos: 3119
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3141
type: B, layer: 1, pos: 3254
type: B, layer: 1, pos: 3299
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3539
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 2980

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2577933, upper bound: 0.2578444
time: 51.21 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2577914, upper bound: 0.2577889
time: 469.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 528.32 seconds
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 528.32
Output dim: 1, lower bound: -0.2577376, upper bound: 0.2562381
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 528.32
Output dim: 1, lower bound: -0.2577367, upper bound: 0.2566441
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 528.32
Output dim: 1, lower bound: -0.2574295, upper bound: 0.2578426
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 528.32
Output dim: 1, lower bound: -0.2574250, upper bound: 0.2577883
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 528.32
Output dim: 1, lower bound: -0.2577933, upper bound: 0.2578444
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 528.32
Output dim: 1, lower bound: -0.2577914, upper bound: 0.2577889

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 97.48 + 3621.33 = 3718.80 seconds
