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
execution time: IAR + LP analysis = 5.83 + 191.51 = 197.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.4199584, upper bound: 0.4199560


# Binary Search by BASE starts (time budget: 17802.67 seconds, max iter: 100)

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
Binary search time: 1325.26 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Relational Split (RS_random_Z) starts
Time budget: 16477.41 seconds

## Binary search (step 0) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 3447

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2658

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3396163, upper bound: 0.3395948
time: 40.06 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3395950, upper bound: 0.3396171
time: 76.50 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 116.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 116.58
Output dim: 1, lower bound: -0.3396163, upper bound: 0.3395948
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 116.58
Output dim: 1, lower bound: -0.3395950, upper bound: 0.3396171

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4471797, 0.4471798
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2844161, 1.2844143
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6745094, 1.6745096
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5862911, 1.5863078
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7280500, 1.7279686
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7548935, 1.7548980
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6230072, 1.6229997
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2747166, 3.2747228
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0862865, 1.0862825
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6305469, 1.6305522

Time for backsubstitution: 2.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 723

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3390845, upper bound: 0.3390311
time: 164.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3390593, upper bound: 0.3390608
time: 63.04 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4471798, 0.4471797
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2844144, 1.2844162
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6745096, 1.6745093
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5863078, 1.5862912
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7279685, 1.7280501
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7548978, 1.7548937
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6229995, 1.6230072
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2747228, 3.2747166
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0862825, 1.0862865
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6305522, 1.6305472

Time for backsubstitution: 2.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 3567

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3395730, upper bound: 0.3395696
time: 53.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3395450, upper bound: 0.3395919
time: 485.14 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 541.68 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 541.68
Output dim: 1, lower bound: -0.3390845, upper bound: 0.3390311
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 541.68
Output dim: 1, lower bound: -0.3390593, upper bound: 0.3390608
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 541.68
Output dim: 1, lower bound: -0.3395730, upper bound: 0.3395696
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 541.68
Output dim: 1, lower bound: -0.3395450, upper bound: 0.3395919

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4471797, 0.4471798
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2844161, 1.2844144
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6745039, 1.6745067
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5862768, 1.5862900
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7280464, 1.7279683
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7548728, 1.7548773
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6229824, 1.6229671
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2747169, 3.2747228
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0862864, 1.0862824
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6305472, 1.6305523

Time for backsubstitution: 3.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2645

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 593

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3390318, upper bound: 0.3390263
time: 428.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3390769, upper bound: 0.3389804
time: 257.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4471797, 0.4471799
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2844161, 1.2844144
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6745063, 1.6745043
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5862734, 1.5862931
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7280498, 1.7279650
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7548730, 1.7548772
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6229748, 1.6229745
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2747169, 3.2747228
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0862864, 1.0862825
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6305472, 1.6305523

Time for backsubstitution: 3.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 3551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2627

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3381003, upper bound: 0.3381034
time: 92.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3381003, upper bound: 0.3381034
time: 95.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4470972, 0.4470958
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2847106, 1.2847259
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6742942, 1.6742924
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5846395, 1.5847309
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7280560, 1.7281367
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7531192, 1.7532285
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6217160, 1.6218176
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2743244, 3.2743149
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0862815, 1.0862856
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6305567, 1.6305516

Time for backsubstitution: 3.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2558

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3028

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3395671, upper bound: 0.3395515
time: 129.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3395524, upper bound: 0.3395632
time: 201.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4470959, 0.4470971
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2847244, 1.2847123
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6742928, 1.6742940
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5847473, 1.5846230
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7280550, 1.7281376
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7532330, 1.7531148
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6218102, 1.6217234
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2743211, 3.2743185
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0862815, 1.0862856
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6305565, 1.6305516

Time for backsubstitution: 3.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2579

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 408

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3384007, upper bound: 0.3394674
time: 69.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3394152, upper bound: 0.3384363
time: 62.81 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 135.79 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 135.79
Output dim: 1, lower bound: -0.3390318, upper bound: 0.3390263
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 135.79
Output dim: 1, lower bound: -0.3390769, upper bound: 0.3389804
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 135.79
Output dim: 1, lower bound: -0.3381003, upper bound: 0.3381034
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 135.79
Output dim: 1, lower bound: -0.3381003, upper bound: 0.3381034
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 135.79
Output dim: 1, lower bound: -0.3395671, upper bound: 0.3395515
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 135.79
Output dim: 1, lower bound: -0.3395524, upper bound: 0.3395632
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 135.79
Output dim: 1, lower bound: -0.3384007, upper bound: 0.3394674
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 135.79
Output dim: 1, lower bound: -0.3394152, upper bound: 0.3384363

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4472110, 0.4472103
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2844558, 1.2844532
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6718588, 1.6716417
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5862191, 1.5862278
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7242266, 1.7238733
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7548274, 1.7548292
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6222212, 1.6222235
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2720926, 3.2719438
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0845790, 1.0846825
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6304969, 1.6304984

Time for backsubstitution: 2.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2723

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3389203, upper bound: 0.3390173
time: 80.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3389204, upper bound: 0.3389076
time: 34.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4472103, 0.4472110
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2844548, 1.2844541
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6716390, 1.6718616
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5862145, 1.5862324
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7239512, 1.7241485
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7548248, 1.7548319
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6222384, 1.6222062
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2719378, 3.2720985
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0846866, 1.0845749
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6304933, 1.6305020

Time for backsubstitution: 2.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2325

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3390574, upper bound: 0.3389600
time: 481.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3390575, upper bound: 0.3389586
time: 385.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4471795, 0.4471796
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2843971, 1.2843684
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6745063, 1.6745044
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5860555, 1.5862517
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.7280400, 1.7279533
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.7546561, 1.7548161
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.6228983, 1.6229125
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2746704, 3.2746840
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0862848, 1.0862808
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6305450, 1.6305482

Time for backsubstitution: 3.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3482

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2654

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3380979, upper bound: 0.3380969
time: 619.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3380979, upper bound: 0.3380988
time: 531.70 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 1154.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1154.07
Output dim: 1, lower bound: -0.3389203, upper bound: 0.3390173
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1154.07
Output dim: 1, lower bound: -0.3389204, upper bound: 0.3389076
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1154.07
Output dim: 1, lower bound: -0.3390574, upper bound: 0.3389600
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1154.07
Output dim: 1, lower bound: -0.3390575, upper bound: 0.3389586
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1154.07
Output dim: 1, lower bound: -0.3380979, upper bound: 0.3380969
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1154.07
Output dim: 1, lower bound: -0.3380979, upper bound: 0.3380988
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1154.07
Output dim: 1, lower bound: -0.3381003, upper bound: 0.3381034
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1154.07
Output dim: 1, lower bound: -0.3395671, upper bound: 0.3395515
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1154.07
Output dim: 1, lower bound: -0.3395524, upper bound: 0.3395632
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1154.07
Output dim: 1, lower bound: -0.3384007, upper bound: 0.3394674
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1154.07
Output dim: 1, lower bound: -0.3394152, upper bound: 0.3384363
Binary search (step 0): status=Status.UNKNOWN, k_low=2, k_high=8, k_mid=5, eps_mid=0.0195312, abs_max=1.2854982614517212
rel_dist={1: [-0.3396559042077183, 0.33965503665474267]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 724

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2382

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853721, upper bound: 0.2853685
time: 95.02 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853748, upper bound: 0.2853756
time: 33.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 128.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 128.38
Output dim: 1, lower bound: -0.2853721, upper bound: 0.2853685
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 128.38
Output dim: 1, lower bound: -0.2853748, upper bound: 0.2853756

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4270553, 0.4270513
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2627385, 1.2627317
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6442370, 1.6442466
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5111201, 1.5114019
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6735895, 1.6736550
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6779683, 1.6782540
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.5291935, 1.5293182
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2168586, 3.2171071
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0631325, 1.0631347
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6182077, 1.6182075

Time for backsubstitution: 3.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 611

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853527, upper bound: 0.2853573
time: 509.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853656, upper bound: 0.2853495
time: 50.63 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4270513, 0.4270553
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2627316, 1.2627386
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6442466, 1.6442370
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5114019, 1.5111202
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6736548, 1.6735897
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6782541, 1.6779680
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.5293185, 1.5291934
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2171071, 3.2168586
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0631347, 1.0631326
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6182075, 1.6182075

Time for backsubstitution: 3.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 3340

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 636

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853321, upper bound: 0.2853319
time: 624.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853321, upper bound: 0.2853341
time: 394.95 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1022.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1022.28
Output dim: 1, lower bound: -0.2853527, upper bound: 0.2853573
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1022.28
Output dim: 1, lower bound: -0.2853656, upper bound: 0.2853495
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1022.28
Output dim: 1, lower bound: -0.2853321, upper bound: 0.2853319
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1022.28
Output dim: 1, lower bound: -0.2853321, upper bound: 0.2853341

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4269747, 0.4269599
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2626834, 1.2626774
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6440492, 1.6440575
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5106006, 1.5109366
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6733778, 1.6734413
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6774342, 1.6777759
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.5281287, 1.5283632
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2167861, 3.2170386
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0631301, 1.0631320
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6181991, 1.6182005

Time for backsubstitution: 3.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2545

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2848291, upper bound: 0.2848209
time: 245.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2848206, upper bound: 0.2848341
time: 71.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4269638, 0.4269708
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2626843, 1.2626765
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6440482, 1.6440585
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5106549, 1.5108821
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6733758, 1.6734433
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6774900, 1.6777201
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.5282387, 1.5282533
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2167900, 3.2170348
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0631299, 1.0631322
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6182005, 1.6181993

Time for backsubstitution: 3.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2722

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853251, upper bound: 0.2853093
time: 327.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853265, upper bound: 0.2853055
time: 241.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4270258, 0.4270523
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2627350, 1.2627369
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6442392, 1.6442343
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5113821, 1.5111059
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6736546, 1.6735892
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6782391, 1.6779574
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.5292693, 1.5291474
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2171061, 3.2168572
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0631340, 1.0631318
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6182050, 1.6182034

Time for backsubstitution: 3.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 2515

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 570

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853253, upper bound: 0.2853291
time: 40.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2853263, upper bound: 0.2853291
time: 910.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4270483, 0.4270298
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2627298, 1.2627419
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6442437, 1.6442298
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.5113876, 1.5111004
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6736546, 1.6735893
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6782434, 1.6779530
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.5292722, 1.5291442
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.2171056, 3.2168577
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0631337, 1.0631318
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6182033, 1.6182051

Time for backsubstitution: 3.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3551

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2852402, upper bound: 0.2852398
time: 66.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2852348, upper bound: 0.2853275
time: 399.47 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 468.89 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 468.89
Output dim: 1, lower bound: -0.2848291, upper bound: 0.2848209
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 468.89
Output dim: 1, lower bound: -0.2848206, upper bound: 0.2848341
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 468.89
Output dim: 1, lower bound: -0.2853251, upper bound: 0.2853093
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 468.89
Output dim: 1, lower bound: -0.2853265, upper bound: 0.2853055
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 468.89
Output dim: 1, lower bound: -0.2853253, upper bound: 0.2853291
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 468.89
Output dim: 1, lower bound: -0.2853263, upper bound: 0.2853291
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 468.89
Output dim: 1, lower bound: -0.2852402, upper bound: 0.2852398
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 468.89
Output dim: 1, lower bound: -0.2852348, upper bound: 0.2853275
Binary search (step 1): status=Status.UNKNOWN, k_low=2, k_high=4, k_mid=3, eps_mid=0.0117188, abs_max=1.2628133296966553
rel_dist={1: [-0.2853804858856417, 0.28537378025840754]}

## Binary search (step 2) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2568
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 3253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2568

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579948, upper bound: 0.2579929
time: 296.13 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579888, upper bound: 0.2579933
time: 106.65 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 402.79 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 402.79
Output dim: 1, lower bound: -0.2579948, upper bound: 0.2579929
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 402.79
Output dim: 1, lower bound: -0.2579888, upper bound: 0.2579933

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4170905, 0.4170899
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2511122, 1.2511131
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6294478, 1.6294482
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.4810133, 1.4809290
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6481502, 1.6481384
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6491668, 1.6490779
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.4861333, 1.4861112
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.1944604, 3.1943436
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0515494, 1.0515497
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6118053, 1.6118069

Time for backsubstitution: 2.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2447

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2743

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579116, upper bound: 0.2579856
time: 446.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579928, upper bound: 0.2579114
time: 27.44 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4170900, 0.4170905
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2511129, 1.2511121
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6294483, 1.6294476
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.4809291, 1.4810134
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6481383, 1.6481500
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6490777, 1.6491671
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.4861109, 1.4861335
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.1943431, 3.1944609
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0515494, 1.0515494
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6118070, 1.6118052

Time for backsubstitution: 3.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 2091

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 422

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2576173, upper bound: 0.2579824
time: 185.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579820, upper bound: 0.2576089
time: 609.43 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 798.11 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 798.11
Output dim: 1, lower bound: -0.2579116, upper bound: 0.2579856
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 798.11
Output dim: 1, lower bound: -0.2579928, upper bound: 0.2579114
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 798.11
Output dim: 1, lower bound: -0.2576173, upper bound: 0.2579824
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 798.11
Output dim: 1, lower bound: -0.2579820, upper bound: 0.2576089

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4170860, 0.4170850
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2509916, 1.2510066
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6294453, 1.6294456
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.4810081, 1.4809252
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6481144, 1.6480963
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6491621, 1.6490746
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.4861183, 1.4860984
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.1944578, 3.1943402
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0515492, 1.0515494
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6117905, 1.6117938

Time for backsubstitution: 3.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2578955, upper bound: 0.2579873
time: 345.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579096, upper bound: 0.2579738
time: 188.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4170855, 0.4170854
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2510056, 1.2509925
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6294453, 1.6294458
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.4810097, 1.4809237
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6481078, 1.6481029
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6491637, 1.6490730
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.4861209, 1.4860959
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.1944573, 3.1943402
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0515492, 1.0515494
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6117922, 1.6117921

Time for backsubstitution: 3.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 3276
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 422
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 672

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3243

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579859, upper bound: 0.2578955
time: 299.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2579784, upper bound: 0.2579086
time: 51.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.7613925, 0.3492787, -0.7613925, 0.3492787, -0.4136877, 0.4136028
1: -0.8265238, 0.9890503, -0.8265238, 0.9890503, -1.2427979, 1.2430055
2: -2.7041636, -0.1659784, -2.7041636, -0.1659784, -1.6231129, 1.6229986
3: -7.1971846, -3.3857131, -7.1971846, -3.3857131, -1.4555879, 1.4550635
4: -6.0309582, -2.7082672, -6.0309582, -2.7082672, -1.6504831, 1.6505811
5: -7.5052495, -3.6067977, -7.5052495, -3.6067977, -1.6272662, 1.6268592
6: -10.5238886, -5.7908278, -10.5238886, -5.7908278, -1.4482044, 1.4473073
7: -6.7266016, -1.9406552, -6.7266016, -1.9406552, -3.1896648, 3.1899006
8: -1.2038996, 0.5512756, -1.2038996, 0.5512756, -1.0504036, 1.0504363
9: -1.4500368, 0.4697701, -1.4500368, 0.4697701, -1.6140586, 1.6140068

Time for backsubstitution: 3.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2663
type: RSZ, layer: 1, pos: 2347
type: RSZ, layer: 1, pos: 2725
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 3253
type: RSZ, layer: 1, pos: 3322
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 3001
type: RSZ, layer: 1, pos: 2302
type: RSZ, layer: 1, pos: 2658
type: RSZ, layer: 1, pos: 3087
type: RSZ, layer: 1, pos: 2466
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 3264
type: RSZ, layer: 1, pos: 2445
type: RSZ, layer: 1, pos: 2737
type: RSZ, layer: 1, pos: 3130
type: RSZ, layer: 1, pos: 2080
type: RSZ, layer: 1, pos: 2405
type: RSZ, layer: 1, pos: 2514
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 3034
type: RSZ, layer: 1, pos: 2430
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 3545
type: RSZ, layer: 1, pos: 2274
type: RSZ, layer: 1, pos: 2558
type: RSZ, layer: 1, pos: 2467
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 2545
type: RSZ, layer: 1, pos: 2338
type: RSZ, layer: 1, pos: 2776
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 2091
type: RSZ, layer: 1, pos: 2967
type: RSZ, layer: 1, pos: 3134
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 3270
type: RSZ, layer: 1, pos: 3566
type: RSZ, layer: 1, pos: 2642
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 51
type: RSZ, layer: 1, pos: 2671
type: RSZ, layer: 1, pos: 2641
type: RSZ, layer: 1, pos: 2214
type: RSZ, layer: 1, pos: 2104
type: RSZ, layer: 1, pos: 2691
type: RSZ, layer: 1, pos: 296
type: RSZ, layer: 1, pos: 2587
type: RSZ, layer: 1, pos: 2738
type: RSZ, layer: 1, pos: 3552
type: RSZ, layer: 1, pos: 3028
type: RSZ, layer: 1, pos: 3299
type: RSZ, layer: 1, pos: 3567
type: RSZ, layer: 1, pos: 2147
type: RSZ, layer: 1, pos: 2412
type: RSZ, layer: 1, pos: 2723
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 2960
type: RSZ, layer: 1, pos: 2134
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 449
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 2762
type: RSZ, layer: 1, pos: 3562
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 3551
type: RSZ, layer: 1, pos: 2337
type: RSZ, layer: 1, pos: 3119
type: RSZ, layer: 1, pos: 2742
type: RSZ, layer: 1, pos: 3223
type: RSZ, layer: 1, pos: 2741
type: RSZ, layer: 1, pos: 3136
type: RSZ, layer: 1, pos: 2273
type: RSZ, layer: 1, pos: 2631
type: RSZ, layer: 1, pos: 3550
type: RSZ, layer: 1, pos: 2105
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 3285
type: RSZ, layer: 1, pos: 2199
type: RSZ, layer: 1, pos: 3473
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 2077
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 180
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 3482
type: RSZ, layer: 1, pos: 3274
type: RSZ, layer: 1, pos: 3533
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 3447
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 3553
type: RSZ, layer: 1, pos: 2078
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 408
type: RSZ, layer: 1, pos: 361
type: RSZ, layer: 1, pos: 2543
type: RSZ, layer: 1, pos: 2424
type: RSZ, layer: 1, pos: 2103
type: RSZ, layer: 1, pos: 2212
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 2974
type: RSZ, layer: 1, pos: 2999
type: RSZ, layer: 1, pos: 3343
type: RSZ, layer: 1, pos: 2213
type: RSZ, layer: 1, pos: 3129
type: RSZ, layer: 1, pos: 3275
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 3054
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 2359
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 2099
type: RSZ, layer: 1, pos: 2681
type: RSZ, layer: 1, pos: 2153
type: RSZ, layer: 1, pos: 2616
type: RSZ, layer: 1, pos: 2560
type: RSZ, layer: 1, pos: 2372
type: RSZ, layer: 1, pos: 3071
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 2175
type: RSZ, layer: 1, pos: 3337
type: RSZ, layer: 1, pos: 3114
type: RSZ, layer: 1, pos: 3563
type: RSZ, layer: 1, pos: 2530
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 2216
type: RSZ, layer: 1, pos: 3582
type: RSZ, layer: 1, pos: 3323
type: RSZ, layer: 1, pos: 2284
type: RSZ, layer: 1, pos: 2516
type: RSZ, layer: 1, pos: 487
type: RSZ, layer: 1, pos: 2980
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 2429
type: RSZ, layer: 1, pos: 2654
type: RSZ, layer: 1, pos: 3243
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 3125
type: RSZ, layer: 1, pos: 2401
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 3252
type: RSZ, layer: 1, pos: 2699
type: RSZ, layer: 1, pos: 2696
type: RSZ, layer: 1, pos: 3141
type: RSZ, layer: 1, pos: 2117
type: RSZ, layer: 1, pos: 2155
type: RSZ, layer: 1, pos: 3278
type: RSZ, layer: 1, pos: 2582
type: RSZ, layer: 1, pos: 2170
type: RSZ, layer: 1, pos: 3316
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 2589
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 2547
type: RSZ, layer: 1, pos: 2395
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 2325
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 3470
type: RSZ, layer: 1, pos: 2169
type: RSZ, layer: 1, pos: 2667
type: RSZ, layer: 1, pos: 2759
type: RSZ, layer: 1, pos: 3245
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 2588
type: RSZ, layer: 1, pos: 3070
type: RSZ, layer: 1, pos: 2441
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 3487
type: RSZ, layer: 1, pos: 2518
type: RSZ, layer: 1, pos: 2461
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 3539
type: RSZ, layer: 1, pos: 2303
type: RSZ, layer: 1, pos: 2743
type: RSZ, layer: 1, pos: 3128
type: RSZ, layer: 1, pos: 2404
type: RSZ, layer: 1, pos: 3340
type: RSZ, layer: 1, pos: 3254
type: RSZ, layer: 1, pos: 2410
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 3088
type: RSZ, layer: 1, pos: 3058
type: RSZ, layer: 1, pos: 3006
type: RSZ, layer: 1, pos: 2341
type: RSZ, layer: 1, pos: 2382
type: RSZ, layer: 1, pos: 2675
type: RSZ, layer: 1, pos: 220
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 87
type: RSZ, layer: 1, pos: 2422
type: RSZ, layer: 1, pos: 2548
type: RSZ, layer: 1, pos: 419
type: RSZ, layer: 1, pos: 2222
type: RSZ, layer: 1, pos: 2627
type: RSZ, layer: 1, pos: 2447
type: RSZ, layer: 1, pos: 2184
type: RSZ, layer: 1, pos: 2573
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 3426
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 2460
type: RSZ, layer: 1, pos: 2669
type: RSZ, layer: 1, pos: 2515
type: RSZ, layer: 1, pos: 3598
type: RSZ, layer: 1, pos: 2146
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 2113
type: RSZ, layer: 1, pos: 2383
type: RSZ, layer: 1, pos: 3021
type: RSZ, layer: 1, pos: 3260
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 2468
type: RSZ, layer: 1, pos: 2984
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 2198
type: RSZ, layer: 1, pos: 2740
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 2572
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 3127
type: RSZ, layer: 1, pos: 2775
type: RSZ, layer: 1, pos: 3519
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 3577
type: RSZ, layer: 1, pos: 2223
type: RSZ, layer: 1, pos: 2272
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 2677
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 2355
type: RSZ, layer: 1, pos: 2676
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 2988
type: RSZ, layer: 1, pos: 2092
type: RSZ, layer: 1, pos: 2722
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 2531
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 3118
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 2562
type: RSZ, layer: 1, pos: 2465
type: RSZ, layer: 1, pos: 3277
type: RSZ, layer: 1, pos: 3306
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 2437
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 3030
type: RSZ, layer: 1, pos: 2626
type: RSZ, layer: 1, pos: 2724
type: RSZ, layer: 1, pos: 3292
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 2158
type: RSZ, layer: 1, pos: 281
type: RSZ, layer: 1, pos: 3291
type: RSZ, layer: 1, pos: 2076
type: RSZ, layer: 1, pos: 151
type: RSZ, layer: 1, pos: 3231
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 3237
type: RSZ, layer: 1, pos: 3544
type: RSZ, layer: 1, pos: 3027
type: RSZ, layer: 1, pos: 2398
type: RSZ, layer: 1, pos: 2235
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 2612
type: RSZ, layer: 1, pos: 143
type: RSZ, layer: 1, pos: 3202
type: RSZ, layer: 1, pos: 2316
type: RSZ, layer: 1, pos: 2579
type: RSZ, layer: 1, pos: 2293
type: RSZ, layer: 1, pos: 3374
type: RSZ, layer: 1, pos: 2576
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 2315
type: RSZ, layer: 1, pos: 2684
type: RSZ, layer: 1, pos: 2448
type: RSZ, layer: 1, pos: 2739
type: RSZ, layer: 1, pos: 2991
type: RSZ, layer: 1, pos: 3325
type: RSZ, layer: 1, pos: 2421
type: RSZ, layer: 1, pos: 2381
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 2597
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 73
type: RSZ, layer: 1, pos: 2645
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 2987
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 2378
type: RSZ, layer: 1, pos: 2397
type: RSZ, layer: 1, pos: 2686
type: RSZ, layer: 1, pos: 3276

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2663

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2576043, upper bound: 0.2576093
time: 450.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2576121, upper bound: 0.2579635
time: 271.85 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 725.80 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 725.80
Output dim: 1, lower bound: -0.2578955, upper bound: 0.2579873
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 725.80
Output dim: 1, lower bound: -0.2579096, upper bound: 0.2579738
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 725.80
Output dim: 1, lower bound: -0.2579859, upper bound: 0.2578955
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 725.80
Output dim: 1, lower bound: -0.2579784, upper bound: 0.2579086
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 725.80
Output dim: 1, lower bound: -0.2576043, upper bound: 0.2576093
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 725.80
Output dim: 1, lower bound: -0.2576121, upper bound: 0.2579635
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 725.80
Output dim: 1, lower bound: -0.2579820, upper bound: 0.2576089
Binary search (step 2): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.2514708042144775
rel_dist={1: [-0.2580117567799336, 0.2580088093270344]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 12980.28 seconds
