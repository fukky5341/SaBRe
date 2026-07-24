## Execution arguments:
Dataset: Dataset.CIFAR10
Network: onnx/cifar10_conv_exp.onnx
Epsilon: 0.03125
Initial delta epsilon: 8
Time budget: 18000 seconds
Threshold: 0.2578837936
Search space: {k/256.0 | k = 1, 2, ..., 8}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4770402, 0.4770402)
1: (-0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.3195257, 1.3195257)
2: (-2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.7195452, 1.7195454)
3: (-7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.6867208, 1.6867208)
4: (-6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.8075225, 1.8075228)
5: (-7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.8555335, 1.8555336)
6: (-10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.7592809, 1.7592809)
7: (-6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.3521972, 3.3521967)
8: (-1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.1210401, 1.1210401)
9: (-1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6491457, 1.6491456)

## BASE Result
execution time: IAR + LP analysis = 4.00 + 193.11 = 197.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.4199584, upper bound: 0.4199560


# Binary Search by BASE starts (time budget: 17802.89 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=8, k_mid=4, eps_mid=0.0156250, abs_max=1.2741557359695435
rel_dist={1: [-0.3125407404242977, 0.312541920538409]}

## Binary search (step 1) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=3, k_mid=2, eps_mid=0.0078125, abs_max=1.2514708042144775
rel_dist={1: [-0.2580117567799336, 0.2580088093270344]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=1, k_mid=1, eps_mid=0.0039062, abs_max=1.2401282787322998
rel_dist={1: [-0.23047414875754169, 0.23047082593412016]}

## Binary Search Result
Binary search time: 1316.54 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Relational Split (RS_dual_Z) starts
Time budget: 16486.35 seconds

## Binary search (step 0) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3550

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3396564, upper bound: 0.3393607
time: 108.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3393616, upper bound: 0.3396515
time: 292.88 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 401.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 401.76
Output dim: 1, lower bound: -0.3396564, upper bound: 0.3393607
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 401.76
Output dim: 1, lower bound: -0.3393616, upper bound: 0.3396515

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4471197, 0.4471192
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2854856, 1.2854848
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6745369, 1.6745381
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5865103, 1.5865130
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7283890, 1.7283916
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7542932, 1.7542948
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6235732, 1.6235806
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2754273, 3.2754273
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0862736, 1.0862739
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6305792, 1.6305789

Time for backsubstitution: 3.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3447

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3394963, upper bound: 0.3391963
time: 547.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3394933, upper bound: 0.3392023
time: 77.72 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4471192, 0.4471197
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2854848, 1.2854855
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6745384, 1.6745367
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5865129, 1.5865103
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7283914, 1.7283893
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7542946, 1.7542932
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6235806, 1.6235733
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2754278, 3.2754269
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0862738, 1.0862737
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6305788, 1.6305791

Time for backsubstitution: 3.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3447

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3391981, upper bound: 0.3394976
time: 42.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3392027, upper bound: 0.3394958
time: 250.88 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 297.21 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 297.21
Output dim: 1, lower bound: -0.3394963, upper bound: 0.3391963
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 297.21
Output dim: 1, lower bound: -0.3394933, upper bound: 0.3392023
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 297.21
Output dim: 1, lower bound: -0.3391981, upper bound: 0.3394976
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 297.21
Output dim: 1, lower bound: -0.3392027, upper bound: 0.3394958

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4471197, 0.4471192
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2854843, 1.2854846
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6745369, 1.6745383
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5865101, 1.5865133
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7283890, 1.7283895
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7542932, 1.7542948
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6235729, 1.6235809
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2754269, 3.2754273
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0862733, 1.0862739
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6305792, 1.6305789

Time for backsubstitution: 3.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3551

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3394961, upper bound: 0.3390447
time: 323.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3393383, upper bound: 0.3391993
time: 96.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4471196, 0.4471192
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2854855, 1.2854848
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6745369, 1.6745383
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5865103, 1.5865129
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7283890, 1.7283915
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7542932, 1.7542948
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6235732, 1.6235805
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2754269, 3.2754273
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0862736, 1.0862739
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6305792, 1.6305789

Time for backsubstitution: 3.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3551

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3394961, upper bound: 0.3390438
time: 349.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3393386, upper bound: 0.3391980
time: 67.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4471192, 0.4471196
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2854835, 1.2854855
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6745384, 1.6745368
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5865129, 1.5865104
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7283914, 1.7283871
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7542946, 1.7542932
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6235803, 1.6235737
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2754273, 3.2754269
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0862736, 1.0862737
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6305788, 1.6305791

Time for backsubstitution: 3.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3551

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3390499, upper bound: 0.3393328
time: 216.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3390460, upper bound: 0.3394953
time: 36.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4471192, 0.4471197
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2854847, 1.2854855
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6745384, 1.6745367
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5865129, 1.5865101
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7283914, 1.7283891
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7542946, 1.7542932
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6235806, 1.6235731
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2754273, 3.2754269
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0862738, 1.0862737
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6305788, 1.6305791

Time for backsubstitution: 3.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3551

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3390496, upper bound: 0.3393321
time: 203.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3390452, upper bound: 0.3394966
time: 46.05 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 253.48 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 253.48
Output dim: 1, lower bound: -0.3394961, upper bound: 0.3390447
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 253.48
Output dim: 1, lower bound: -0.3393383, upper bound: 0.3391993
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 253.48
Output dim: 1, lower bound: -0.3394961, upper bound: 0.3390438
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 253.48
Output dim: 1, lower bound: -0.3393386, upper bound: 0.3391980
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 253.48
Output dim: 1, lower bound: -0.3390499, upper bound: 0.3393328
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 253.48
Output dim: 1, lower bound: -0.3390460, upper bound: 0.3394953
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 253.48
Output dim: 1, lower bound: -0.3390496, upper bound: 0.3393321
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 253.48
Output dim: 1, lower bound: -0.3390452, upper bound: 0.3394966

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4470999, 0.4470987
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2855064, 1.2855039
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6745709, 1.6745758
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5865600, 1.5865654
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7284191, 1.7284219
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7543399, 1.7543441
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6235914, 1.6236029
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2754464, 3.2754474
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0862734, 1.0862739
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6306083, 1.6306045

Time for backsubstitution: 3.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3566

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3394295, upper bound: 0.3390056
time: 224.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3394295, upper bound: 0.3390073
time: 244.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4470991, 0.4470994
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2855035, 1.2855067
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6745743, 1.6745723
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5865624, 1.5865635
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7284212, 1.7284198
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7543423, 1.7543416
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6235950, 1.6235993
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2754474, 3.2754464
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0862734, 1.0862739
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6306050, 1.6306078

Time for backsubstitution: 3.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3566

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3393034, upper bound: 0.3391340
time: 311.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3393034, upper bound: 0.3391337
time: 787.97 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 1102.70 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1102.70
Output dim: 1, lower bound: -0.3394295, upper bound: 0.3390056
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1102.70
Output dim: 1, lower bound: -0.3394295, upper bound: 0.3390073
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1102.70
Output dim: 1, lower bound: -0.3393034, upper bound: 0.3391340
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1102.70
Output dim: 1, lower bound: -0.3393034, upper bound: 0.3391337
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1102.70
Output dim: 1, lower bound: -0.3394961, upper bound: 0.3390438
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1102.70
Output dim: 1, lower bound: -0.3393386, upper bound: 0.3391980
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1102.70
Output dim: 1, lower bound: -0.3390499, upper bound: 0.3393328
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1102.70
Output dim: 1, lower bound: -0.3390460, upper bound: 0.3394953
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1102.70
Output dim: 1, lower bound: -0.3390496, upper bound: 0.3393321
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1102.70
Output dim: 1, lower bound: -0.3390452, upper bound: 0.3394966
Binary search (step 0): status=Status.UNKNOWN, k_low=2, k_high=8, k_mid=5, eps_mid=0.0195312, abs_max=1.2854982614517212
rel_dist={1: [-0.3396559042077183, 0.33965503665474267]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3550

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853710, upper bound: 0.2851963
time: 233.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2851933, upper bound: 0.2853740
time: 586.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 820.24 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 820.24
Output dim: 1, lower bound: -0.2853710, upper bound: 0.2851963
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 820.24
Output dim: 1, lower bound: -0.2851933, upper bound: 0.2853740

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4271717, 0.4271714
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2628003, 1.2627997
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6445208, 1.6445215
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5196828, 1.5196846
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6755531, 1.6755549
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6867905, 1.6867914
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.5331434, 1.5331478
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2242603, 3.2242603
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0631109, 1.0631112
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6181984, 1.6181980

Time for backsubstitution: 3.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3447

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853024, upper bound: 0.2851431
time: 231.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853195, upper bound: 0.2851219
time: 497.53 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4271715, 0.4271717
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2627997, 1.2628002
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6445215, 1.6445205
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5196847, 1.5196829
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6755545, 1.6755534
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6867917, 1.6867905
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.5331479, 1.5331435
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2242608, 3.2242603
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0631112, 1.0631111
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6181982, 1.6181985

Time for backsubstitution: 3.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3447

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2851261, upper bound: 0.2853180
time: 477.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2851407, upper bound: 0.2853041
time: 73.89 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 554.89 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 554.89
Output dim: 1, lower bound: -0.2853024, upper bound: 0.2851431
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 554.89
Output dim: 1, lower bound: -0.2853195, upper bound: 0.2851219
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 554.89
Output dim: 1, lower bound: -0.2851261, upper bound: 0.2853180
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 554.89
Output dim: 1, lower bound: -0.2851407, upper bound: 0.2853041

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4271717, 0.4271714
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2627995, 1.2627997
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6445208, 1.6445216
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5196828, 1.5196846
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6755531, 1.6755536
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6867905, 1.6867915
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.5331432, 1.5331481
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2242599, 3.2242599
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0631107, 1.0631112
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6181984, 1.6181980

Time for backsubstitution: 3.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3551

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853071, upper bound: 0.2850460
time: 395.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2852112, upper bound: 0.2851401
time: 33.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4271717, 0.4271714
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2628002, 1.2627997
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6445208, 1.6445215
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5196828, 1.5196843
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6755531, 1.6755548
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6867905, 1.6867914
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.5331434, 1.5331477
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2242599, 3.2242603
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0631109, 1.0631112
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6181984, 1.6181980

Time for backsubstitution: 3.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3551

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853209, upper bound: 0.2850329
time: 61.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2852253, upper bound: 0.2851256
time: 606.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4271715, 0.4271716
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2627990, 1.2628002
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6445215, 1.6445206
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5196844, 1.5196829
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6755545, 1.6755521
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6867914, 1.6867906
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.5331477, 1.5331436
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2242603, 3.2242599
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0631109, 1.0631111
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6181982, 1.6181985

Time for backsubstitution: 3.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3551

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2851265, upper bound: 0.2852246
time: 211.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2850358, upper bound: 0.2853208
time: 60.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4271714, 0.4271717
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2627997, 1.2628002
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6445215, 1.6445206
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5196847, 1.5196826
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6755545, 1.6755533
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6867917, 1.6867905
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.5331479, 1.5331433
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2242603, 3.2242603
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0631109, 1.0631111
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6181982, 1.6181985

Time for backsubstitution: 3.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3551

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2851424, upper bound: 0.2852075
time: 141.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2850505, upper bound: 0.2853024
time: 66.95 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 211.63 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 211.63
Output dim: 1, lower bound: -0.2853071, upper bound: 0.2850460
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 211.63
Output dim: 1, lower bound: -0.2852112, upper bound: 0.2851401
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 211.63
Output dim: 1, lower bound: -0.2853209, upper bound: 0.2850329
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 211.63
Output dim: 1, lower bound: -0.2852253, upper bound: 0.2851256
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 211.63
Output dim: 1, lower bound: -0.2851265, upper bound: 0.2852246
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 211.63
Output dim: 1, lower bound: -0.2850358, upper bound: 0.2853208
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 211.63
Output dim: 1, lower bound: -0.2851424, upper bound: 0.2852075
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 211.63
Output dim: 1, lower bound: -0.2850505, upper bound: 0.2853024

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4271517, 0.4271509
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2628216, 1.2628200
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6445562, 1.6445591
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5197337, 1.5197368
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6755841, 1.6755859
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6868384, 1.6868408
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.5331631, 1.5331700
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2242799, 3.2242801
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0631108, 1.0631112
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6182275, 1.6182253

Time for backsubstitution: 3.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3566

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2852711, upper bound: 0.2850256
time: 598.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2852711, upper bound: 0.2850237
time: 345.21 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 947.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 947.21
Output dim: 1, lower bound: -0.2852711, upper bound: 0.2850256
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 947.21
Output dim: 1, lower bound: -0.2852711, upper bound: 0.2850237
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 947.21
Output dim: 1, lower bound: -0.2852112, upper bound: 0.2851401
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 947.21
Output dim: 1, lower bound: -0.2853209, upper bound: 0.2850329
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 947.21
Output dim: 1, lower bound: -0.2852253, upper bound: 0.2851256
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 947.21
Output dim: 1, lower bound: -0.2851265, upper bound: 0.2852246
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 947.21
Output dim: 1, lower bound: -0.2850358, upper bound: 0.2853208
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 947.21
Output dim: 1, lower bound: -0.2851424, upper bound: 0.2852075
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 947.21
Output dim: 1, lower bound: -0.2850505, upper bound: 0.2853024
Binary search (step 1): status=Status.UNKNOWN, k_low=2, k_high=4, k_mid=3, eps_mid=0.0117188, abs_max=1.2628133296966553
rel_dist={1: [-0.2853804858856417, 0.28537378025840754]}

## Binary search (step 2) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3550

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2580067, upper bound: 0.2578922
time: 92.15 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2578938, upper bound: 0.2580104
time: 48.06 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 140.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 140.38
Output dim: 1, lower bound: -0.2580067, upper bound: 0.2578922
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 140.38
Output dim: 1, lower bound: -0.2578938, upper bound: 0.2580104

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4171978, 0.4171976
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2514577, 1.2514572
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6295125, 1.6295131
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.4862691, 1.4862703
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6491354, 1.6491364
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6530392, 1.6530399
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.4879285, 1.4879315
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.1986771, 3.1986771
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0515295, 1.0515299
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6120079, 1.6120077

Time for backsubstitution: 3.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3447

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579679, upper bound: 0.2578746
time: 24.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579915, upper bound: 0.2578433
time: 250.87 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4171976, 0.4171978
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2514572, 1.2514577
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6295133, 1.6295125
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.4862703, 1.4862692
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6491363, 1.6491355
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6530399, 1.6530392
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.4879314, 1.4879286
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.1986771, 3.1986766
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0515298, 1.0515296
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6120079, 1.6120080

Time for backsubstitution: 3.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3447

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2578473, upper bound: 0.2579918
time: 54.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2578705, upper bound: 0.2579669
time: 51.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 110.03 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 110.03
Output dim: 1, lower bound: -0.2579679, upper bound: 0.2578746
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 110.03
Output dim: 1, lower bound: -0.2579915, upper bound: 0.2578433
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 110.03
Output dim: 1, lower bound: -0.2578473, upper bound: 0.2579918
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 110.03
Output dim: 1, lower bound: -0.2578705, upper bound: 0.2579669

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4171977, 0.4171975
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2514570, 1.2514572
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6295125, 1.6295131
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.4862691, 1.4862703
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6491354, 1.6491355
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6530392, 1.6530399
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.4879283, 1.4879316
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.1986766, 3.1986766
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0515295, 1.0515298
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6120079, 1.6120077

Time for backsubstitution: 3.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3551

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579680, upper bound: 0.2578085
time: 46.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579019, upper bound: 0.2578700
time: 41.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4171977, 0.4171976
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2514575, 1.2514572
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6295125, 1.6295131
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.4862691, 1.4862701
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6491354, 1.6491363
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6530392, 1.6530399
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.4879285, 1.4879314
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.1986766, 3.1986771
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0515295, 1.0515299
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6120079, 1.6120077

Time for backsubstitution: 3.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3551

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579907, upper bound: 0.2577868
time: 42.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579261, upper bound: 0.2578477
time: 885.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4171976, 0.4171977
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2514567, 1.2514575
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6295133, 1.6295125
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.4862701, 1.4862691
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6491363, 1.6491345
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6530399, 1.6530393
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.4879314, 1.4879286
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.1986766, 3.1986761
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0515295, 1.0515296
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6120079, 1.6120080

Time for backsubstitution: 3.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3551

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2578503, upper bound: 0.2579220
time: 410.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2577864, upper bound: 0.2579791
time: 409.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4171976, 0.4171978
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2514572, 1.2514577
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6295133, 1.6295125
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.4862703, 1.4862690
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6491363, 1.6491354
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6530399, 1.6530392
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.4879314, 1.4879284
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.1986766, 3.1986766
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0515295, 1.0515296
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6120079, 1.6120080

Time for backsubstitution: 3.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 3551

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2578728, upper bound: 0.2579015
time: 68.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2578110, upper bound: 0.2579683
time: 46.68 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 118.74 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 118.74
Output dim: 1, lower bound: -0.2579680, upper bound: 0.2578085
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 118.74
Output dim: 1, lower bound: -0.2579019, upper bound: 0.2578700
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 118.74
Output dim: 1, lower bound: -0.2579907, upper bound: 0.2577868
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 118.74
Output dim: 1, lower bound: -0.2579261, upper bound: 0.2578477
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 118.74
Output dim: 1, lower bound: -0.2578503, upper bound: 0.2579220
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 118.74
Output dim: 1, lower bound: -0.2577864, upper bound: 0.2579791
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 118.74
Output dim: 1, lower bound: -0.2578728, upper bound: 0.2579015
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 118.74
Output dim: 1, lower bound: -0.2578110, upper bound: 0.2579683

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4171775, 0.4171770
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2514790, 1.2514780
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6295489, 1.6295507
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.4863203, 1.4863224
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6491666, 1.6491679
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6530876, 1.6530892
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.4879489, 1.4879535
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.1986961, 3.1986969
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0515296, 1.0515298
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6120372, 1.6120355

Time for backsubstitution: 3.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3566

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579462, upper bound: 0.2577973
time: 35.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579462, upper bound: 0.2577973
time: 35.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4171772, 0.4171773
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2514780, 1.2514791
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6295503, 1.6295494
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.4863212, 1.4863216
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6491675, 1.6491671
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6530886, 1.6530882
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.4879503, 1.4879521
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.1986966, 3.1986964
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0515296, 1.0515298
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6120358, 1.6120369

Time for backsubstitution: 3.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3598

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 3566

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2578913, upper bound: 0.2578469
time: 518.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2578913, upper bound: 0.2578483
time: 522.93 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 1044.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1044.80
Output dim: 1, lower bound: -0.2579462, upper bound: 0.2577973
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1044.80
Output dim: 1, lower bound: -0.2579462, upper bound: 0.2577973
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1044.80
Output dim: 1, lower bound: -0.2578913, upper bound: 0.2578469
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1044.80
Output dim: 1, lower bound: -0.2578913, upper bound: 0.2578483
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1044.80
Output dim: 1, lower bound: -0.2579907, upper bound: 0.2577868
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1044.80
Output dim: 1, lower bound: -0.2579261, upper bound: 0.2578477
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1044.80
Output dim: 1, lower bound: -0.2578503, upper bound: 0.2579220
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1044.80
Output dim: 1, lower bound: -0.2577864, upper bound: 0.2579791
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1044.80
Output dim: 1, lower bound: -0.2578728, upper bound: 0.2579015
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1044.80
Output dim: 1, lower bound: -0.2578110, upper bound: 0.2579683
Binary search (step 2): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.2514708042144775
rel_dist={1: [-0.2580117567799336, 0.2580088093270344]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 13795.34 seconds
